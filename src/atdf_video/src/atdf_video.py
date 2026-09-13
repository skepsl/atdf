#!/usr/bin/env python3
"""
V4_ATDF_NRP_target_finding.py

Target-finding implementation using the trained Neural Ray Predictor (NRP) as
ATDF's forward model. This ROS-enabled version supports the real robot loop:
move_base navigation goals are issued for the robot base, the robot pose is
read from the navigation/localization stack (TF map->base_link by default, with
/amcl_pose as a fallback), a rigid transform converts the base pose to the
side-mounted antenna pose, and real BLE ISP1907 IQ snapshots replace the
previous Sionna measurement. The default simulation launch instead uses Isaac
ground truth at stopped measurement poses and an independent Sionna worker.
RViz retains measurement and source-estimate histories plus the current RF PF.

NRP input convention:
    u = [tx_x, tx_y, rx_x, rx_y]
normalized with the same map bounds used during data generation.

NRP / Sionna ray tuple convention:
    [norm_gain, norm_delay, AoD_theta, AoD_phi, AoA_theta, AoA_phi]
where theta/phi follow the Sionna convention: theta=zenith, phi=azimuth.

Requested debug limits remain enforced internally:
    T_max <= 100, N_p <= 1000.
"""

import glob
import json
import math
import os
# ROS nodes must not use a GUI Matplotlib backend. TkAgg can crash in ROS
# callback/action threads with Tcl_AsyncDelete. Force a file-only backend.
os.environ.setdefault("MPLBACKEND", "Agg")
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from atdf_artifacts import RunArtifacts
from atdf_plot import make_reference_plot

try:
    import rospy
    import rostopic
    import actionlib
    from actionlib_msgs.msg import GoalID, GoalStatus
    from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    from std_msgs.msg import Header, String
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    rostopic = None
    actionlib = None
    GoalStatus = None
    PoseStamped = None
    PoseArray = None
    Pose = None
    PoseWithCovarianceStamped = None
    Odometry = None
    TFMessage = None
    MoveBaseAction = None
    MoveBaseGoal = None
    Header = None
    String = None
    euler_from_quaternion = None
    quaternion_from_euler = None
    ROS_AVAILABLE = False


try:
    import tf
except Exception:
    tf = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # fallback is used only if scipy is unavailable
    distance_transform_edt = None

class Tester:
    """Compatibility adapter for the independent Sionna Python worker."""

    def __init__(self, args=None):
        from atdf_sionna import SionnaClient
        self.client = SionnaClient(
            database_dir=str(_ros_param("~sionna_database_dir", "~/atdf_database")),
            timeout_s=float(_ros_param("~sionna_timeout_s", 120.0)),
            is_shutdown=(rospy.is_shutdown if ROS_AVAILABLE else None),
        )

    def inquire_ray(self, rx_batch_2d_np, tx_batch_2d_np):
        return self.client.inquire_ray(rx_batch_2d_np, tx_batch_2d_np)


class RealIQDataAcquisition:
    """
    On-demand BLE ISP1907 IQ collector used by ATDF on the real robot.

    This is intentionally not the same as iq.IQDataAcquisition's timer-driven
    node.  ATDF must collect IQ only after move_base has reached a goal, so this
    class subscribes to the same String topic and exposes collect_processed_iq()
    as a blocking measurement call.
    """
    def __init__(
        self,
        topic: str = "isp_ble/data",
        device: str = "cpu",
        selected_indices: Optional[Sequence[int]] = None,
        queue_maxlen: int = 1000000,
        warmup_s: float = 1.5,
        timeout_s: float = 60.0,
        extra_frames: int = 5,
    ) -> None:
        if not ROS_AVAILABLE or rospy is None or String is None:
            raise RuntimeError("RealIQDataAcquisition requires ROS and std_msgs/String")
        self.topic = str(topic)
        self.device = str(device)
        self.selected_indices = list(selected_indices if selected_indices is not None else [7, 8, 9, 10])
        self.warmup_s = float(warmup_s)
        self.timeout_s = float(timeout_s)
        self.extra_frames = int(max(0, extra_frames))
        self.antenna_data_frame = deque(maxlen=int(queue_maxlen))
        self._sub = rospy.Subscriber(self.topic, String, self.antenna_callback, queue_size=200)
        rospy.loginfo(f"[ATDF] Real IQ acquisition subscribed to /{self.topic.lstrip('/')}; selected_indices={self.selected_indices}")

    def antenna_callback(self, msg) -> None:
        try:
            data_list = json.loads(msg.data)
            self.antenna_data_frame.append(data_list)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, f"[ATDF] Failed to parse IQ antenna message: {exc}")

    def rf_signal_calibration(self, package):
        iq_data = []
        for item in package:
            try:
                if item[0] == "IQ" and item[3] != 255:
                    iq_data.append(item)
            except Exception:
                continue

        complex_data = []
        for item in iq_data:
            try:
                complex_data.append(float(item[4]) + 1j * float(item[5]))
            except Exception:
                continue

        complex_data = np.asarray(complex_data, dtype=np.complex128)
        if len(complex_data) != 22:
            return None

        ref_ts = np.arange(8, dtype=float) * 0.000001
        meas_ts = ref_ts[-1] + 0.000002 + np.arange(14) * 0.000002
        n = np.concatenate([ref_ts, meas_ts])

        phi_time = 2.0 * np.pi * (-250000.0) * n
        complex_data = complex_data * np.exp(-1j * phi_time)

        ref_iqs = complex_data[:8]
        T = 1e-6
        phase_diff = np.angle(ref_iqs[1:] * np.conj(ref_iqs[:-1]))
        avg_phase_diff = np.mean(phase_diff)
        f_est = avg_phase_diff / (2.0 * np.pi * T)

        phi_time = 2.0 * np.pi * f_est * n
        corrected_iqs = complex_data * np.exp(-1j * phi_time)

        phi_meas_updated = np.angle(corrected_iqs)
        cal_iq = np.asarray(corrected_iqs * np.exp(-1j * phi_meas_updated[0]), dtype=np.complex64)
        return torch.as_tensor(cal_iq, dtype=torch.complex64, device=self.device)

    def selecting_rf_data(self, calibrated_measurements: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
        iq_ula = []
        for calibrated_data in calibrated_measurements:
            if calibrated_data is None:
                continue
            if int(calibrated_data.numel()) <= max(self.selected_indices):
                continue
            above = torch.stack([calibrated_data[int(i)] for i in self.selected_indices]).to(self.device)
            if int(above.numel()) == len(self.selected_indices):
                iq_ula.append(above)
        if len(iq_ula) == 0:
            return None
        return torch.stack(iq_ula, dim=0)  # [num_measurement, num_antenna]

    def collect_processed_iq(self, num_measurements: int = 20) -> Optional[torch.Tensor]:
        num_measurements = int(max(1, num_measurements))
        rospy.loginfo(f"[ATDF] Begin real IQ collection: requested={num_measurements}")
        if self.warmup_s > 0.0:
            rospy.sleep(self.warmup_s)

        self.antenna_data_frame.clear()
        start_time = time.time()
        current_data_len = 0
        required = num_measurements + self.extra_frames
        while len(self.antenna_data_frame) < required and not rospy.is_shutdown():
            rospy.sleep(0.1)
            if len(self.antenna_data_frame) > current_data_len:
                current_data_len = len(self.antenna_data_frame)
                start_time = time.time()
            if time.time() - start_time > self.timeout_s:
                rospy.logwarn(f"[ATDF] IQ data collection timeout after {self.timeout_s:.1f}s; frames={len(self.antenna_data_frame)}")
                break

        frames = list(self.antenna_data_frame)
        calibrated = []
        for frame in frames:
            c = self.rf_signal_calibration(frame)
            if c is not None:
                calibrated.append(c)
            if len(calibrated) >= num_measurements:
                break

        iq_processed = self.selecting_rf_data(calibrated)
        n_valid = 0 if iq_processed is None else int(iq_processed.shape[0])
        rospy.loginfo(f"[ATDF] IQ collection complete: raw_frames={len(frames)}, valid_measurements={n_valid}")
        return iq_processed


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class AdaLN(nn.Module):
    def __init__(self, d_model, d_context):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(d_context, d_model * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, z_env):
        gamma_beta = self.linear(z_env).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * self.norm(x) + beta


class ContextTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_context, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.adaLN_1 = AdaLN(d_model, d_context)
        self.adaLN_2 = AdaLN(d_model, d_context)

    def forward(self, x, z_env):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.adaLN_1(x + attn_out, z_env)
        ffn_out = self.ffn(x)
        x = self.adaLN_2(x + ffn_out, z_env)
        return x


class NeuralRayPredictor(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.num_queries = int(cfg_model['num_queries'])
        d_model = int(cfg_model['d_model'])
        self.backbone = nn.Sequential(
            SineLayer(4, d_model, is_first=True, omega_0=cfg_model['omega_0_first']),
            SineLayer(d_model, d_model, is_first=False, omega_0=cfg_model['omega_0_hidden']),
            SineLayer(d_model, d_model, is_first=False, omega_0=cfg_model['omega_0_hidden']),
            nn.Linear(d_model, d_model),
        )
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.decoder_layers = nn.ModuleList([
            ContextTransformerLayer(d_model, int(cfg_model['nhead']), d_model)
            for _ in range(int(cfg_model['num_layers']))
        ])
        self.head_prob = nn.Linear(d_model, 1)
        self.head_gain = nn.Linear(d_model, 1)
        self.head_delay = nn.Linear(d_model, 1)
        self.head_angle = nn.Linear(d_model, 4)

    def predict_from_queries(self, q):
        logits = self.head_prob(q)
        p_k = torch.sigmoid(logits)
        gain = torch.sigmoid(self.head_gain(q))
        delay = torch.sigmoid(self.head_delay(q))
        angles = self.head_angle(q)
        r_k = torch.cat([gain, delay, angles], dim=-1)
        return logits, p_k, r_k

    def forward(self, u, return_aux=True):
        B = u.shape[0]
        z_env = self.backbone(u)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        aux_outputs = []
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            q = layer(q, z_env)
            if return_aux and (num_layers - 4 <= i < num_layers - 1):
                aux_outputs.append(self.predict_from_queries(q))
        logits, p_k, r_k = self.predict_from_queries(q)
        if return_aux:
            return logits, p_k, r_k, aux_outputs
        return logits, p_k, r_k


class ATDF:
    """
    Active Target Detection / Finding with a trained NRP forward model.

    The state is a weighted particle approximation of the hidden Tx position in
    the map frame. The robot/Rx state is [x, y, yaw]. The NRP consumes only
    [tx_x, tx_y, rx_x, rx_y]; yaw is used by the receiver-array covariance
    surrogate and by the planning tie-breaks. Sionna RT can optionally remain as
    the simulated measurement source for validation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        map_yaml_path: str = "maps/21202.yaml",
        checkpoint_path: Optional[str] = "checkpoints/epoch_0771.pth",
        device: Optional[str] = None,
        valid_pool_stride_px: int = 6,
        rt_rx_chunk_size: int = 8,
        rt_tx_chunk_size: int = 128,
        nrp_batch_size: int = 512,
        forward_model: str = "nrp",
        measurement_source: str = "sionna",
        restrict_to_nrp_domain: Optional[bool] = None,
        nrp_input_clamp: Optional[bool] = None,
        antenna_offset_x_m: Optional[float] = None,
        antenna_offset_y_m: Optional[float] = None,
        antenna_yaw_offset_rad: Optional[float] = None,
        snap_forward_rx_to_valid: Optional[bool] = None,
        real_iq_topic: str = "isp_ble/data",
        real_iq_num_measurements: Optional[int] = None,
        real_iq_selected_indices: Optional[Sequence[int]] = None,
        real_iq_negative_frequency: Optional[bool] = None,
        real_iq_reverse_array: Optional[bool] = None,
        real_iq_average_mode: Optional[str] = None,
        real_iq_normalize_snapshots: Optional[bool] = None,
        real_iq_covariance_comparison: Optional[str] = None,
        real_iq_cov_sigma: Optional[float] = None,
        real_iq_phase_offsets_rad: Optional[Sequence[float]] = None,
    ) -> None:

        self.rand_idx = f"{np.random.randint(0, 9999):04d}"
        print(f"RUN ID: {self.rand_idx}")

        config_path = self._resolve_path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_model = str(forward_model).lower().strip()
        self.measurement_source = str(measurement_source).lower().strip()
        if self.forward_model == "rt":
            self.forward_model = "sionna"
        if self.measurement_source == "rt":
            self.measurement_source = "sionna"
        if self.measurement_source in {"real", "iq", "ble", "ble_iq"}:
            self.measurement_source = "real_iq"
        if self.forward_model not in {"nrp", "sionna"}:
            raise ValueError("forward_model must be 'nrp' or 'sionna'")
        if self.measurement_source not in {"nrp", "sionna", "real_iq"}:
            raise ValueError("measurement_source must be 'nrp', 'sionna', or 'real_iq'")

        self.nrp_batch_size = int(max(1, nrp_batch_size))
        self.rt_rx_chunk_size = int(rt_rx_chunk_size)
        self.rt_tx_chunk_size = int(rt_tx_chunk_size)

        # External Sionna ray inquiry helper is now needed only when Sionna is
        # used as a measurement source or as an explicit baseline forward model.
        needs_sionna = (self.forward_model == "sionna") or (self.measurement_source == "sionna")
        if needs_sionna:
            if Tester is None:
                raise ImportError("utils.Engine3D.Tester is required for Sionna measurement/forward mode")
            self.tester = Tester(args=None)
        else:
            self.tester = None

        # Normalization parameters must match atf_testing.py / training data.
        self.P_min = float(self.cfg["dataset"].get("P_min", -120.0))
        self.P_max = float(self.cfg["dataset"].get("P_max", -40.0))
        self.tau_min = float(self.cfg["dataset"].get("tau_min", 0.0))
        self.tau_max = float(self.cfg["dataset"].get("tau_max", 1e-6))

        # Dataset bounds are kept only for optional normalization/debugging.
        self.map_x_min = float(self.cfg.get("atdf", {}).get("map_x_min", -13.0))
        self.map_x_max = float(self.cfg.get("atdf", {}).get("map_x_max", 6.0))
        self.map_y_min = float(self.cfg.get("atdf", {}).get("map_y_min", -2.5))
        self.map_y_max = float(self.cfg.get("atdf", {}).get("map_y_max", 10.0))

        eval_cfg = self.cfg.get("eval", {})
        pat_cfg = self.cfg.get("rf", {}).get("rx_pattern", {})

        # Real BLE ISP1907 IQ measurement options. In real-robot mode, the
        # measurement source is a complex 4-antenna IQ vector/covariance instead
        # of Sionna ray tuples. The forward model should normally remain NRP.
        self.real_iq_topic = str(eval_cfg.get("real_iq_topic", real_iq_topic))
        self.real_iq_num_measurements = int(eval_cfg.get("real_iq_num_measurements", 20) if real_iq_num_measurements is None else real_iq_num_measurements)
        if real_iq_selected_indices is None:
            real_iq_selected_indices = eval_cfg.get("real_iq_selected_indices", [7, 8, 9, 10])
        self.real_iq_selected_indices = [int(v) for v in list(real_iq_selected_indices)]
        self.real_iq_negative_frequency = bool(eval_cfg.get("real_iq_negative_frequency", True) if real_iq_negative_frequency is None else real_iq_negative_frequency)
        self.real_iq_reverse_array = bool(eval_cfg.get("real_iq_reverse_array", False) if real_iq_reverse_array is None else real_iq_reverse_array)
        self.real_iq_average_mode = str(eval_cfg.get("real_iq_average_mode", "mean_vector") if real_iq_average_mode is None else real_iq_average_mode).lower().strip()
        self.real_iq_normalize_snapshots = bool(eval_cfg.get("real_iq_normalize_snapshots", True) if real_iq_normalize_snapshots is None else real_iq_normalize_snapshots)
        self.real_iq_covariance_comparison = str(eval_cfg.get("real_iq_covariance_comparison", "correlation") if real_iq_covariance_comparison is None else real_iq_covariance_comparison).lower().strip()
        self.real_iq_cov_sigma = float(eval_cfg.get("real_iq_cov_sigma", 0.50) if real_iq_cov_sigma is None else real_iq_cov_sigma)
        if real_iq_phase_offsets_rad is None:
            real_iq_phase_offsets_rad = eval_cfg.get("real_iq_phase_offsets_rad", [0.0, 0.0, 0.0, 0.0])
        self.real_iq_phase_offsets_rad = [float(v) for v in list(real_iq_phase_offsets_rad)]
        if len(self.real_iq_phase_offsets_rad) != 4:
            self.real_iq_phase_offsets_rad = [0.0, 0.0, 0.0, 0.0]
        self.real_iq_use_robust_likelihood = bool(eval_cfg.get("real_iq_use_robust_likelihood", True))
        self.real_iq_trace_normalize = bool(eval_cfg.get("real_iq_trace_normalize", True))
        self.real_iq_min_valid_measurements = int(eval_cfg.get("real_iq_min_valid_measurements", max(1, min(4, self.real_iq_num_measurements))))
        self.real_iq_timeout_s = float(eval_cfg.get("real_iq_timeout_s", 60.0))
        self.real_iq_warmup_s = float(eval_cfg.get("real_iq_warmup_s", 1.5))
        self.real_iq_extra_frames = int(eval_cfg.get("real_iq_extra_frames", 5))
        self.iq_receiver: Optional[RealIQDataAcquisition] = None
        self._last_real_iq: Optional[torch.Tensor] = None
        self._last_real_iq_mean: Optional[torch.Tensor] = None
        self._last_real_iq_cov: Optional[torch.Tensor] = None
        self._last_real_iq_stats: Dict[str, float] = {}
        if self.measurement_source == "real_iq":
            self.iq_receiver = RealIQDataAcquisition(
                topic=self.real_iq_topic,
                device=self.device,
                selected_indices=self.real_iq_selected_indices,
                warmup_s=self.real_iq_warmup_s,
                timeout_s=self.real_iq_timeout_s,
                extra_frames=self.real_iq_extra_frames,
            )

        # Rigid antenna mounting model. ROS/base_link convention is x-forward,
        # y-left, yaw=0 along +x. The antenna is on the robot right side, so the
        # default lateral offset is negative y. Its local z/lobe axis points to
        # 270 deg when the base yaw is 0, i.e. yaw offset = -pi/2.
        self.antenna_offset_x_m = float(
            eval_cfg.get("antenna_offset_x_m", 0.0)
            if antenna_offset_x_m is None
            else antenna_offset_x_m
        )
        self.antenna_offset_y_m = float(
            eval_cfg.get("antenna_offset_y_m", -0.30)
            if antenna_offset_y_m is None
            else antenna_offset_y_m
        )
        self.antenna_yaw_offset_rad = float(
            eval_cfg.get("antenna_yaw_offset_rad", -0.5 * math.pi)
            if antenna_yaw_offset_rad is None
            else antenna_yaw_offset_rad
        )
        # In ROS/Isaac mode, the RF query pose is the physical antenna point,
        # not the Scout base centre. Snapping it to the robot-centre valid pool
        # can silently corrupt the measurement coordinate, so the default is off.
        self.snap_forward_rx_to_valid = bool(
            eval_cfg.get("snap_forward_rx_to_valid", False)
            if snap_forward_rx_to_valid is None
            else snap_forward_rx_to_valid
        )

        # ROS runtime state. These are initialized lazily by setup_ros_interfaces().
        # Real robot mode must not depend on ground truth.  The robot pose comes
        # from localization: TF map->base_link by default, with /amcl_pose or
        # another configured pose topic as a fallback.
        self._ros_ready = False
        self._latest_robot_pose = None
        self._latest_robot_pose_stamp = None
        self._latest_robot_sample = None
        self._robot_pose_sub = None
        self._tf_listener = None
        self.allow_goal_pose_fallback = False
        # Backward-compatible aliases used only by older simulation/debug code.
        self._latest_gt_pose = None
        self._latest_gt_stamp = None
        self._gt_sub = None
        self._goal_pub = None
        self._goal_marker_pub = None
        self._estimate_pub = None
        self._antenna_pose_pub = None
        self._particle_pub = None
        self._move_base_client = None
        self._last_goal_pose = None
        self._rviz_markers = None
        self._latest_motion = None
        self._latest_command = None
        self._measurement_stamp = None

        # NRP options. The model was trained on normalized [tx_x, tx_y, rx_x, rx_y]
        # coordinates. Restricting valid Tx/Rx pools to this domain prevents the
        # PF/planner from asking the NRP to extrapolate far outside training data.
        self.p_threshold = float(eval_cfg.get("p_threshold", 0.5))
        self.nrp_ray_top_k = int(eval_cfg.get("nrp_ray_top_k", self.cfg.get("dataset", {}).get("max_rays", 20)))
        self.nrp_cov_top_k = int(eval_cfg.get("nrp_cov_top_k", 24))
        self.nrp_prob_gamma = float(eval_cfg.get("nrp_prob_gamma", 1.0))
        self.nrp_input_clamp = bool(eval_cfg.get("nrp_input_clamp", True)) if nrp_input_clamp is None else bool(nrp_input_clamp)
        self.restrict_to_nrp_domain = (
            bool(eval_cfg.get("restrict_to_nrp_domain", self.forward_model == "nrp"))
            if restrict_to_nrp_domain is None
            else bool(restrict_to_nrp_domain)
        )

        # Initial-belief support.  By default the prior is uniform over every
        # valid Tx point in the active validity pool.  You can restrict it with
        # init_prior_bounds=[xmin, xmax, ymin, ymax] in config or by passing
        # init_bounds/init_x_range/init_y_range to run_active_localization().
        self.default_init_prior_bounds = self._parse_xy_bounds(eval_cfg.get("init_prior_bounds", None))
        self.default_init_x_range = self._parse_range(eval_cfg.get("init_prior_x_range", None))
        self.default_init_y_range = self._parse_range(eval_cfg.get("init_prior_y_range", None))
        self.print_prior_debug = bool(eval_cfg.get("print_prior_debug", True))

        self.model: Optional[NeuralRayPredictor] = None
        self.checkpoint_path: Optional[str] = None
        if self.forward_model == "nrp" or self.measurement_source == "nrp":
            self._init_nrp_model(checkpoint_path)

        # Receiver pattern / array options.
        self.rx_pat_m = float(pat_cfg.get("m", 4.0))
        self.rx_pat_gmin = float(pat_cfg.get("g_min", 1e-3))
        self.rx_pat_front_only = bool(pat_cfg.get("front_only", True))
        self.aoa_is_propagation = bool(eval_cfg.get("aoa_is_propagation", True))

        # Sionna convention is theta=zenith, phi=azimuth. Set this False only
        # if your Tester intentionally remaps angles before returning them.
        self.sionna_theta_phi = bool(eval_cfg.get("sionna_theta_phi", True))

        # Planning / score settings. These are deliberately conservative so the
        # RT path solver is not called with 1000 particles for every candidate.
        self.mi_tol_abs = float(eval_cfg.get("mi_tol_abs", 1e-3))
        self.mi_tol_rel = float(eval_cfg.get("mi_tol_rel", 0.02))
        self.mi_flat_threshold = float(eval_cfg.get("mi_flat_threshold", 1e-2))
        self.lambda_face = float(eval_cfg.get("lambda_face", 0.35))
        self.lambda_smooth = float(eval_cfg.get("lambda_smooth", 0.10))
        self.lambda_progress = float(eval_cfg.get("lambda_progress", 0.85))
        self.lambda_mi = float(eval_cfg.get("lambda_mi", 1.0))
        self.use_oracle_heading = bool(eval_cfg.get("use_oracle_heading", False))

        # The ROS occupancy map should not be an absolute topological barrier in
        # this Sionna-RT debug mode: Sionna explicitly models refraction and
        # penetration.  We therefore allow a candidate Rx move to cross a short
        # occupied segment when the endpoint is free and the RT/planning score
        # justifies paying a quantitative wall-crossing cost.
        self.allow_wall_crossing = bool(eval_cfg.get("allow_wall_crossing", True))
        self.wall_cross_max_step = float(eval_cfg.get("wall_cross_max_step", 3.60))
        self.wall_cross_probe_count = int(eval_cfg.get("wall_cross_probe_count", 33))
        self.wall_cross_max_occupied_len = float(eval_cfg.get("wall_cross_max_occupied_len", 1.10))
        self.wall_cross_max_unknown_len = float(eval_cfg.get("wall_cross_max_unknown_len", 0.35))
        self.wall_cross_max_segments = int(eval_cfg.get("wall_cross_max_segments", 2))
        self.wall_cross_max_contig_occupied_len = float(eval_cfg.get("wall_cross_max_contig_occupied_len", 0.85))
        self.wall_cross_unknown_weight = float(eval_cfg.get("wall_cross_unknown_weight", 0.50))
        self.wall_cross_segment_penalty = float(eval_cfg.get("wall_cross_segment_penalty", 0.15))
        self.wall_cross_cost_norm = float(
            eval_cfg.get(
                "wall_cross_cost_norm",
                max(0.25, self.wall_cross_max_occupied_len + self.wall_cross_segment_penalty),
            )
        )
        self.lambda_wall_cost = float(eval_cfg.get("lambda_wall_cost", 0.45))
        self.lambda_penetration_support = float(eval_cfg.get("lambda_penetration_support", 0.65))
        self.wall_cross_min_score_margin = float(eval_cfg.get("wall_cross_min_score_margin", 0.00))

        # Candidate expansion for wall penetration.  The Scout footprint can make
        # the first free cell behind a wall infeasible, so we must search deeper
        # and allow several landing poses per direction before the RT-MI stage.
        self.wall_cross_max_landings_per_dir = int(eval_cfg.get("wall_cross_max_landings_per_dir", 3))
        self.wall_cross_landing_snap_radius_m = float(eval_cfg.get("wall_cross_landing_snap_radius_m", 0.85))
        self.wall_cross_landing_snap_k = int(eval_cfg.get("wall_cross_landing_snap_k", 96))
        self.wall_cross_heading_count = int(eval_cfg.get("wall_cross_heading_count", 12))
        self.max_candidate_poses = int(eval_cfg.get("max_candidate_poses", 240))
        self.max_wall_cross_candidates = int(eval_cfg.get("max_wall_cross_candidates", 144))
        raw_range_factors = eval_cfg.get("candidate_range_factors", [1.0, 1.6, 2.3])
        if isinstance(raw_range_factors, (int, float)):
            raw_range_factors = [float(raw_range_factors)]
        self.candidate_range_factors = [float(v) for v in raw_range_factors if float(v) > 0.0]
        if len(self.candidate_range_factors) == 0:
            self.candidate_range_factors = [1.0]

        # Robot-footprint safety. The Rx is mounted on an AgileX Scout Mini
        # sized base, so Rx candidate poses are no longer point-valid only.
        # The planner keeps the robot center away from obstacles and verifies
        # an oriented rectangular footprint for each candidate yaw.
        self.robot_width_m = float(eval_cfg.get("robot_width_m", 0.65))
        self.robot_length_m = float(eval_cfg.get("robot_length_m", 0.65))
        self.robot_safety_margin_m = float(eval_cfg.get("robot_safety_margin_m", 0.08))
        # self.robot_center_clearance_m = float(
        #     eval_cfg.get("robot_center_clearance_m", 0.5 * self.robot_width_m + self.robot_safety_margin_m)
        # )
        self.robot_center_clearance_m = float(
            eval_cfg.get("robot_center_clearance_m", 0.48)
        )
        self.robot_footprint_sample_m = float(eval_cfg.get("robot_footprint_sample_m", 0.04))
        self.clearance_bonus_range_m = float(eval_cfg.get("clearance_bonus_range_m", 0.50))
        self.lambda_clearance = float(eval_cfg.get("lambda_clearance", 0.15))
        self.draw_robot_footprint = bool(eval_cfg.get("draw_robot_footprint", True))
        self._robot_local_footprint_cache: Optional[torch.Tensor] = None

        # Particle-filter likelihood defaults. These can be overridden in
        # run_active_localization().
        self.ray_likelihood_top_k = int(eval_cfg.get("ray_likelihood_top_k", 8))
        default_ray_sigma = 0.45 if self.forward_model == "nrp" else 0.25
        self.ray_likelihood_sigma = float(eval_cfg.get("ray_likelihood_sigma", default_ray_sigma))
        self.ray_count_penalty = float(eval_cfg.get("ray_count_penalty", 0.35))
        self.ray_extra_pred_weight = float(eval_cfg.get("ray_extra_pred_weight", 0.35))
        self.ray_rss_weight = float(eval_cfg.get("ray_rss_weight", 0.25))
        self.likelihood_temperature = float(eval_cfg.get("likelihood_temperature", 1.0))

        # Map / occupancy.
        self._load_ros_map(map_yaml_path)
        self._build_valid_point_pools(stride_px=valid_pool_stride_px)

        # RF / covariance surrogate.
        self._rf_inited = False
        self._init_rf_params()

    # ==================================================================
    # Path, map, and validity helpers
    # ==================================================================
    def _resolve_path(self, path: str, search_dirs: Optional[Sequence[str]] = None) -> str:
        """Resolve project paths and uploaded '(n)' file variants robustly."""
        if path is None:
            raise FileNotFoundError("Path is None")
        if os.path.exists(path):
            return path

        # Resolve a source checkout or a catkin installation without a user's
        # absolute workspace path. Launch files still pass explicit paths.
        package_dirs = [os.path.dirname(os.path.dirname(os.path.realpath(__file__)))]
        try:
            import rospkg
            package_dirs.append(rospkg.RosPack().get_path("atdf_video"))
        except Exception:
            pass
        for directory in package_dirs:
            candidate = os.path.join(directory, path.lstrip("/"))
            if os.path.isfile(candidate):
                return candidate

        search_dirs = list(search_dirs or [])
        base = os.path.basename(path)
        stem, ext = os.path.splitext(base)

        dirs = [os.getcwd(), "/mnt/data", *search_dirs]
        seen = set()
        for d in dirs:
            if not d or d in seen:
                continue
            seen.add(d)
            direct = os.path.join(d, base)
            if os.path.exists(direct):
                return direct
            for cand in sorted(glob.glob(os.path.join(d, f"{stem}*{ext}"))):
                if os.path.exists(cand):
                    return cand

        raise FileNotFoundError(f"Could not resolve path: {path}")

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> str:
        if checkpoint_path is None:
            raise FileNotFoundError("checkpoint_path is None; provide checkpoints/epoch_0771.pth or set measurement_source='sionna' and forward_model='sionna'")
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        # Also try common project-local checkpoint folders and uploaded variants.
        rel_dirs = [
            os.path.join(os.getcwd(), "checkpoints"),
            os.path.join(os.getcwd(), "checkpoint"),
            "/checkpoints",
            "/mnt/data",
        ]
        try:
            return self._resolve_path(checkpoint_path, search_dirs=rel_dirs)
        except FileNotFoundError:
            base = os.path.basename(checkpoint_path)
            for d in rel_dirs:
                cand = os.path.join(d, base)
                if os.path.exists(cand):
                    return cand
            raise

    def _extract_state_dict(self, ckpt_obj) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt_obj, dict):
            for key in ["model_state_dict", "state_dict", "model", "net", "ema", "module"]:
                val = ckpt_obj.get(key, None)
                if isinstance(val, dict):
                    ckpt_obj = val
                    break
        if not isinstance(ckpt_obj, dict):
            raise TypeError("Checkpoint does not contain a state_dict-like object")
        state = {}
        for k, v in ckpt_obj.items():
            if not torch.is_tensor(v):
                continue
            kk = str(k)
            for prefix in ["module.", "model.", "net."]:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            state[kk] = v
        if len(state) == 0:
            raise ValueError("No tensor entries were found in checkpoint state_dict")
        return state

    def _init_nrp_model(self, checkpoint_path: Optional[str]) -> None:
        self.model = NeuralRayPredictor(cfg_model=self.cfg["model"]).to(self.device)
        ckpt_path = self._resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = self._extract_state_dict(ckpt)
        try:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load NRP checkpoint '{ckpt_path}': {exc}") from exc
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[NRP] loaded with missing={len(missing)} unexpected={len(unexpected)} from {ckpt_path}")
            if len(missing) > 0:
                print("[NRP] first missing keys:", list(missing)[:5])
            if len(unexpected) > 0:
                print("[NRP] first unexpected keys:", list(unexpected)[:5])
        else:
            print(f"[NRP] loaded {n_params:,} parameters from {ckpt_path}")

    def within_nrp_domain_mask(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0]
        y = xy[..., 1]
        return (
            (x >= float(self.map_x_min))
            & (x <= float(self.map_x_max))
            & (y >= float(self.map_y_min))
            & (y <= float(self.map_y_max))
        )

    @staticmethod
    def _parse_range(value) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Parse [lo, hi] / {'min': lo, 'max': hi} / None into a numeric range."""
        if value is None:
            return None
        if isinstance(value, str):
            txt = value.replace(",", " ").replace("[", " ").replace("]", " ")
            vals = [v for v in txt.split() if v.lower() not in {"none", "null"}]
            if len(vals) != 2:
                raise ValueError(f"Range string must contain two values, got: {value}")
            lo, hi = float(vals[0]), float(vals[1])
        elif isinstance(value, dict):
            lo = value.get("min", value.get("lo", value.get("lower", None)))
            hi = value.get("max", value.get("hi", value.get("upper", None)))
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
        else:
            vals = list(value)
            if len(vals) != 2:
                raise ValueError(f"Range must contain two values, got: {value}")
            lo = None if vals[0] is None else float(vals[0])
            hi = None if vals[1] is None else float(vals[1])
        if any(v is not None and not math.isfinite(v) for v in (lo, hi)):
            raise ValueError(f"Source PF range limits must be finite, got: {value}")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"Source PF range minimum must be less than maximum, got: {value}")
        return (lo, hi)

    def _parse_xy_bounds(self, value) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
        """Parse [xmin, xmax, ymin, ymax] or {'x':[...], 'y':[...]} into bounds."""
        if value is None:
            return None
        if isinstance(value, dict):
            xr = self._parse_range(value.get("x", value.get("x_range", None)))
            yr = self._parse_range(value.get("y", value.get("y_range", None)))
            xmin, xmax = (None, None) if xr is None else xr
            ymin, ymax = (None, None) if yr is None else yr
            return (xmin, xmax, ymin, ymax)
        vals = list(value)
        if len(vals) != 4:
            raise ValueError(f"init_prior_bounds must be [xmin, xmax, ymin, ymax], got: {value}")
        xmin, xmax = self._parse_range(vals[:2])
        ymin, ymax = self._parse_range(vals[2:])
        return (xmin, xmax, ymin, ymax)

    def combine_prior_bounds(
        self,
        init_bounds=None,
        init_x_range=None,
        init_y_range=None,
    ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
        bounds = self._parse_xy_bounds(init_bounds) if init_bounds is not None else self.default_init_prior_bounds
        # Explicit rectangle limits take precedence over configured axis
        # defaults. Explicit axis arguments can still refine the rectangle.
        xr = self._parse_range(init_x_range) if init_x_range is not None else (
            self.default_init_x_range if init_bounds is None else None
        )
        yr = self._parse_range(init_y_range) if init_y_range is not None else (
            self.default_init_y_range if init_bounds is None else None
        )
        if bounds is None:
            bounds = (None, None, None, None)
        xmin, xmax, ymin, ymax = bounds
        if xr is not None:
            xmin, xmax = xr
        if yr is not None:
            ymin, ymax = yr
        if xmin is None and xmax is None and ymin is None and ymax is None:
            return None
        return self._parse_xy_bounds((xmin, xmax, ymin, ymax))

    def points_in_bounds(self, xy: torch.Tensor, bounds=None) -> torch.Tensor:
        if bounds is None:
            return torch.ones(xy.shape[:-1], dtype=torch.bool, device=xy.device)
        xmin, xmax, ymin, ymax = bounds
        x = xy[..., 0]
        y = xy[..., 1]
        mask = torch.ones_like(x, dtype=torch.bool)
        if xmin is not None:
            mask = mask & (x >= float(xmin))
        if xmax is not None:
            mask = mask & (x <= float(xmax))
        if ymin is not None:
            mask = mask & (y >= float(ymin))
        if ymax is not None:
            mask = mask & (y <= float(ymax))
        return mask

    @staticmethod
    def _bounds_to_tensor(bounds, device: str) -> Optional[torch.Tensor]:
        if bounds is None:
            return None
        vals = [float("nan") if v is None else float(v) for v in bounds]
        return torch.tensor(vals, device=device, dtype=torch.float32)

    @staticmethod
    def _tensor_to_bounds(t: Optional[torch.Tensor]):
        if t is None:
            return None
        vals = t.detach().cpu().reshape(-1).tolist()
        if len(vals) != 4:
            return None
        return tuple(None if math.isnan(float(v)) else float(v) for v in vals)

    @staticmethod
    def format_bounds(bounds) -> str:
        if bounds is None:
            return "full valid-pool"
        names = ["xmin", "xmax", "ymin", "ymax"]
        parts = []
        for name, val in zip(names, bounds):
            parts.append(f"{name}=None" if val is None else f"{name}={float(val):+.2f}")
        return "[" + ", ".join(parts) + "]"

    def warn_if_prior_extrapolates_nrp(self, bounds) -> None:
        if bounds is None or not getattr(self, "restrict_to_nrp_domain", False):
            return
        xmin, xmax, ymin, ymax = bounds
        outside = False
        if xmin is not None and xmin < float(self.map_x_min):
            outside = True
        if xmax is not None and xmax > float(self.map_x_max):
            outside = True
        if ymin is not None and ymin < float(self.map_y_min):
            outside = True
        if ymax is not None and ymax > float(self.map_y_max):
            outside = True
        if outside:
            print(
                "[prior] requested bounds extend outside the NRP training-normalization domain "
                f"x=[{self.map_x_min:+.2f},{self.map_x_max:+.2f}], "
                f"y=[{self.map_y_min:+.2f},{self.map_y_max:+.2f}]. "
                "Because restrict_to_nrp_domain=True, the effective prior is clipped to this domain. "
                "Set eval.restrict_to_nrp_domain=false and eval.nrp_input_clamp=false only if you intentionally accept NRP extrapolation."
            )

    def normalize_nrp_input(self, tx_xy: torch.Tensor, rx_xy: torch.Tensor) -> torch.Tensor:
        tx_xy = tx_xy.to(self.device, dtype=torch.float32).view(-1, 2)
        rx_xy = rx_xy.to(self.device, dtype=torch.float32).view(-1, 2)
        if tx_xy.shape[0] != rx_xy.shape[0]:
            raise ValueError(f"tx_xy and rx_xy must have same batch size, got {tx_xy.shape[0]} and {rx_xy.shape[0]}")
        u = torch.cat([tx_xy, rx_xy], dim=-1)
        x_den = max(float(self.map_x_max - self.map_x_min), 1e-12)
        y_den = max(float(self.map_y_max - self.map_y_min), 1e-12)
        u[:, [0, 2]] = (u[:, [0, 2]] - float(self.map_x_min)) / x_den
        u[:, [1, 3]] = (u[:, [1, 3]] - float(self.map_y_min)) / y_den
        if self.nrp_input_clamp:
            u = torch.clamp(u, 0.0, 1.0)
        return u

    @torch.no_grad()
    def nrp_predict_batch(self, tx_xy: torch.Tensor, rx_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("NRP model is not initialized; set forward_model='nrp' or measurement_source='nrp'")
        tx_xy = tx_xy.to(self.device, dtype=torch.float32).view(-1, 2)
        rx_xy = rx_xy.to(self.device, dtype=torch.float32).view(-1, 2)
        B = int(tx_xy.shape[0])
        p_parts: List[torch.Tensor] = []
        r_parts: List[torch.Tensor] = []
        bs = int(max(1, self.nrp_batch_size))
        for b0 in range(0, B, bs):
            b1 = min(b0 + bs, B)
            u = self.normalize_nrp_input(tx_xy[b0:b1], rx_xy[b0:b1])
            _, p_k, r_k = self.model(u, return_aux=False)
            p_parts.append(p_k.detach())
            r_parts.append(r_k.detach())
        return torch.cat(p_parts, dim=0), torch.cat(r_parts, dim=0)

    def _nrp_ray_mask(self, probs: torch.Tensor, top_k: Optional[int] = None, threshold: Optional[float] = None) -> torch.Tensor:
        probs = probs.reshape(-1).clamp(0.0, 1.0)
        q = int(probs.numel())
        if q == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        k = int(top_k if top_k is not None else self.nrp_ray_top_k)
        k = max(1, min(k, q))
        pth = float(self.p_threshold if threshold is None else threshold)
        mask = probs >= pth
        if int(mask.sum().item()) == 0:
            idx = torch.topk(probs, k=k, largest=True).indices
            mask = torch.zeros(q, dtype=torch.bool, device=self.device)
            mask[idx] = True
        elif int(mask.sum().item()) > k:
            keep_src = torch.where(mask)[0]
            top_local = torch.topk(probs[keep_src], k=k, largest=True).indices
            new_mask = torch.zeros(q, dtype=torch.bool, device=self.device)
            new_mask[keep_src[top_local]] = True
            mask = new_mask
        return mask

    def nrp_outputs_to_ray_list(self, p_k: torch.Tensor, r_k: torch.Tensor, top_k: Optional[int] = None) -> List[torch.Tensor]:
        if p_k.ndim == 3:
            probs = p_k.squeeze(-1)
        else:
            probs = p_k
        B = int(r_k.shape[0])
        out: List[torch.Tensor] = []
        for b in range(B):
            mask = self._nrp_ray_mask(probs[b], top_k=top_k)
            rays = r_k[b, mask, :6]
            if rays.numel() == 0:
                rays = torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            else:
                order = torch.argsort(rays[:, 0], descending=True)
                rays = rays[order]
            out.append(rays.to(self.device, dtype=torch.float32))
        return out

    @torch.no_grad()
    def query_nrp_rays_cross(self, rx_pose_batch: torch.Tensor, tx_xy_batch: torch.Tensor, snap_to_valid: bool = True) -> List[torch.Tensor]:
        if rx_pose_batch.ndim == 1:
            rx_pose_batch = rx_pose_batch.unsqueeze(0)
        if tx_xy_batch.ndim == 1:
            tx_xy_batch = tx_xy_batch.unsqueeze(0)
        rx_pose_batch = rx_pose_batch.to(self.device, dtype=torch.float32).clone()
        tx_xy_batch = tx_xy_batch.to(self.device, dtype=torch.float32).clone()
        if snap_to_valid and getattr(self, "snap_forward_rx_to_valid", False):
            rx_pose_batch[:, :2] = self.snap_points_to_valid(rx_pose_batch[:, :2], role="rx")
        if snap_to_valid:
            tx_xy_batch = self.snap_points_to_valid(tx_xy_batch, role="tx")
        R = int(rx_pose_batch.shape[0])
        T = int(tx_xy_batch.shape[0])
        rx_rep = rx_pose_batch[:, None, :2].expand(R, T, 2).reshape(R * T, 2)
        tx_rep = tx_xy_batch[None, :, :].expand(R, T, 2).reshape(R * T, 2)
        p_k, r_k = self.nrp_predict_batch(tx_rep, rx_rep)
        return self.nrp_outputs_to_ray_list(p_k, r_k, top_k=self.nrp_ray_top_k)

    @torch.no_grad()
    def covariance_from_nrp_outputs(self, rx_pose: torch.Tensor, p_k: torch.Tensor, r_k: torch.Tensor) -> torch.Tensor:
        if rx_pose.ndim == 1:
            rx_pose = rx_pose.unsqueeze(0)
        rx_pose = rx_pose.to(self.device, dtype=torch.float32)
        if p_k.ndim == 3:
            probs = p_k.squeeze(-1)
        else:
            probs = p_k
        probs = probs.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        r_k = r_k.to(self.device, dtype=torch.float32)
        B, Q = probs.shape
        k = int(max(1, min(self.nrp_cov_top_k, Q)))
        top = torch.topk(probs, k=k, dim=1, largest=True).indices
        idx_feat = top.unsqueeze(-1).expand(B, k, r_k.shape[-1])
        r_sel = torch.gather(r_k, dim=1, index=idx_feat)
        p_sel = torch.gather(probs, dim=1, index=top)
        # Use existence probability as a soft path weight. This is smoother than
        # hard p-thresholding and avoids losing weak but informative NLoS paths.
        mask = torch.clamp(p_sel, 0.0, 1.0).pow(float(self.nrp_prob_gamma))
        alpha_bar = r_sel[..., 0]
        aoa_theta_g = r_sel[..., 4]
        aoa_phi_g = r_sel[..., 5]
        return self.covariance_from_rays(rx_pose, alpha_bar, aoa_theta_g, aoa_phi_g, mask)

    @torch.no_grad()
    def covariances_from_nrp_cross(self, rx_pose_batch: torch.Tensor, tx_xy_batch: torch.Tensor) -> torch.Tensor:
        if rx_pose_batch.ndim == 1:
            rx_pose_batch = rx_pose_batch.unsqueeze(0)
        if tx_xy_batch.ndim == 1:
            tx_xy_batch = tx_xy_batch.unsqueeze(0)
        rx_pose_batch = rx_pose_batch.to(self.device, dtype=torch.float32).clone()
        tx_xy_batch = tx_xy_batch.to(self.device, dtype=torch.float32).clone()
        if getattr(self, "snap_forward_rx_to_valid", False):
            rx_pose_batch[:, :2] = self.snap_points_to_valid(rx_pose_batch[:, :2], role="rx")
        tx_xy_batch = self.snap_points_to_valid(tx_xy_batch, role="tx")
        R = int(rx_pose_batch.shape[0])
        T = int(tx_xy_batch.shape[0])
        rx_rep_xy = rx_pose_batch[:, None, :2].expand(R, T, 2).reshape(R * T, 2)
        tx_rep = tx_xy_batch[None, :, :].expand(R, T, 2).reshape(R * T, 2)
        p_k, r_k = self.nrp_predict_batch(tx_rep, rx_rep_xy)
        rx_rep_pose = rx_pose_batch[:, None, :].expand(R, T, 3).reshape(R * T, 3)
        C = self.covariance_from_nrp_outputs(rx_rep_pose, p_k, r_k)
        return C.reshape(R, T, self.N_R, self.N_R)

    @torch.no_grad()
    def query_forward_rays_cross(self, rx_pose_batch: torch.Tensor, tx_xy_batch: torch.Tensor, forward_model: Optional[str] = None) -> List[torch.Tensor]:
        model = str(forward_model or self.forward_model).lower().strip()
        if model == "rt":
            model = "sionna"
        if model == "nrp":
            return self.query_nrp_rays_cross(rx_pose_batch, tx_xy_batch)
        if model == "sionna":
            return self.query_rt_rays_cross(rx_pose_batch, tx_xy_batch)
        raise ValueError("forward_model must be 'nrp' or 'sionna'")

    @torch.no_grad()
    def covariances_from_forward_cross(self, rx_pose_batch: torch.Tensor, tx_xy_batch: torch.Tensor, forward_model: Optional[str] = None) -> torch.Tensor:
        model = str(forward_model or self.forward_model).lower().strip()
        if model == "rt":
            model = "sionna"
        if model == "nrp":
            return self.covariances_from_nrp_cross(rx_pose_batch, tx_xy_batch)
        if model == "sionna":
            return self.covariances_from_rt_cross(rx_pose_batch, tx_xy_batch)
        raise ValueError("forward_model must be 'nrp' or 'sionna'")

    @staticmethod
    def _canonical_measurement_source(src: str) -> str:
        src = str(src).lower().strip()
        if src == "rt":
            return "sionna"
        if src in {"real", "iq", "ble", "ble_iq"}:
            return "real_iq"
        return src

    def _is_covariance_measurement(self, z) -> bool:
        return (
            torch.is_tensor(z)
            and torch.is_complex(z)
            and z.ndim >= 2
            and int(z.shape[-1]) == int(self.N_R)
            and int(z.shape[-2]) == int(self.N_R)
        )

    def _complex_trace_normalize(self, C: torch.Tensor) -> torch.Tensor:
        C = C.to(self.device)
        tr = torch.real(torch.diagonal(C, dim1=-2, dim2=-1)).sum(dim=-1).clamp_min(1e-12)
        return C * (float(self.N_R) / tr[..., None, None])

    def _covariance_to_correlation(self, C: torch.Tensor) -> torch.Tensor:
        C = C.to(self.device)
        diag = torch.real(torch.diagonal(C, dim1=-2, dim2=-1)).clamp_min(1e-12)
        scale = torch.sqrt(diag[..., :, None] * diag[..., None, :]).clamp_min(1e-12)
        return C / scale

    def normalize_covariance_for_real_iq(self, C: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self, "real_iq_covariance_comparison", "correlation")).lower().strip()
        C = self._hermitize(C.to(self.device))
        if mode in {"corr", "correlation", "phase", "phase_only"}:
            return self._hermitize(self._covariance_to_correlation(C))
        if mode in {"trace", "trace_norm", "normalized"}:
            return self._hermitize(self._complex_trace_normalize(C))
        return C

    @torch.no_grad()
    def real_iq_to_observed_covariance(self, iq_processed: torch.Tensor) -> torch.Tensor:
        """
        Convert real BLE ISP1907 IQ snapshots [M,4] to one observed spatial
        covariance [4,4] in the same steering convention as the NRP/Sionna model.

        The BLE baseband is negative-frequency.  For a ULA this conjugates the
        array phase progression, which mirrors left/right AoA if interpreted with
        the positive-frequency Sionna convention.  Therefore the default is to
        conjugate the measured IQ before likelihood evaluation.
        """
        if iq_processed is None:
            raise RuntimeError("No real IQ measurements were collected from the antenna")
        if not torch.is_tensor(iq_processed):
            iq = torch.as_tensor(iq_processed)
        else:
            iq = iq_processed
        iq = iq.to(self.device)
        if not torch.is_complex(iq):
            # Accept a last dimension of [real, imag] if a future driver emits it.
            if iq.ndim >= 2 and int(iq.shape[-1]) == 2 and int(iq.shape[-2]) == self.N_R:
                iq = torch.view_as_complex(iq.to(torch.float32).contiguous())
            else:
                iq = iq.to(torch.complex64)
        else:
            iq = iq.to(torch.complex64)
        if iq.ndim == 1:
            iq = iq.view(1, -1)
        if int(iq.shape[-1]) != int(self.N_R):
            raise ValueError(f"Expected real IQ tensor [M,{self.N_R}], got shape {tuple(iq.shape)}")
        if int(iq.shape[0]) < int(self.real_iq_min_valid_measurements):
            raise RuntimeError(
                f"Only {int(iq.shape[0])} valid IQ measurements were collected; "
                f"minimum is {int(self.real_iq_min_valid_measurements)}"
            )

        if bool(self.real_iq_reverse_array):
            iq = torch.flip(iq, dims=[-1])
        if bool(self.real_iq_negative_frequency):
            iq = torch.conj(iq)

        # Optional per-channel phase calibration for hardware-to-Sionna alignment.
        # Use this after a calibration run with a known transmitter position.
        phase_offsets = torch.as_tensor(self.real_iq_phase_offsets_rad, device=self.device, dtype=torch.float32)
        if int(phase_offsets.numel()) == int(self.N_R):
            iq = iq * torch.exp(-1j * phase_offsets).to(torch.complex64).view(1, -1)

        if bool(self.real_iq_normalize_snapshots):
            rms = torch.sqrt(torch.mean(torch.abs(iq).pow(2), dim=-1, keepdim=True)).clamp_min(1e-12)
            iq = iq / rms

        mode = str(self.real_iq_average_mode).lower().strip()
        if mode in {"sample_cov", "sample_covariance", "cov", "covariance"}:
            C = torch.einsum("mi,mj->ij", iq, torch.conj(iq)) / float(max(1, int(iq.shape[0])))
            y_mean = iq.mean(dim=0)
        else:
            # Requested default: average across all measurement periods to get
            # one stable 4-IQ snapshot, then form a rank-one spatial covariance.
            y_mean = iq.mean(dim=0)
            if bool(self.real_iq_normalize_snapshots):
                y_mean = y_mean / torch.sqrt(torch.mean(torch.abs(y_mean).pow(2))).clamp_min(1e-12)
            C = y_mean[:, None] * torch.conj(y_mean[None, :])

        C = self._hermitize(C)
        if bool(self.real_iq_trace_normalize):
            C = self._complex_trace_normalize(C)
        I = torch.eye(self.N_R, device=self.device, dtype=C.dtype)
        C = self._hermitize(C + 1e-9 * I)

        self._last_real_iq = iq.detach().clone()
        self._last_real_iq_mean = y_mean.detach().clone()
        self._last_real_iq_cov = C.detach().clone()
        self._last_real_iq_stats = {
            "num_iq": float(iq.shape[0]),
            "iq_power_mean": float(torch.mean(torch.abs(iq).pow(2)).item()),
            "iq_mean_abs": float(torch.mean(torch.abs(y_mean)).item()),
            "iq_cov_trace": float(torch.real(torch.diagonal(C)).sum().item()),
        }
        return C

    @torch.no_grad()
    def loglik_from_observed_iq_cov(self, C_obs: torch.Tensor, C_pred: torch.Tensor) -> torch.Tensor:
        if C_obs.ndim == 2:
            C_obs_b = C_obs.unsqueeze(0)
        else:
            C_obs_b = C_obs
        if C_pred.ndim == 2:
            C_pred_b = C_pred.unsqueeze(0)
        else:
            C_pred_b = C_pred
        B = int(C_pred_b.shape[0])
        C_obs_n = self.normalize_covariance_for_real_iq(C_obs_b)[0]
        C_pred_n = self.normalize_covariance_for_real_iq(C_pred_b)
        D = C_pred_n - C_obs_n.unsqueeze(0).expand(B, -1, -1)
        # The diagonal is sensitive to per-chain gain calibration.  Use the
        # off-diagonal complex correlation pattern, which carries AoA phase.
        eye = torch.eye(self.N_R, device=self.device, dtype=torch.bool).unsqueeze(0)
        D = D.masked_fill(eye, 0.0)
        err = (D.real.pow(2) + D.imag.pow(2)).sum(dim=(-1, -2)) / float(max(1, self.N_R * (self.N_R - 1)))
        sigma = max(float(self.real_iq_cov_sigma), 1e-6)
        ll = -err / (2.0 * sigma * sigma)
        return torch.nan_to_num(torch.clamp(ll, min=-80.0, max=0.0), nan=-80.0, posinf=0.0, neginf=-80.0)

    @torch.no_grad()
    def measurement_real_iq_fn(self, rx_pose: Optional[torch.Tensor] = None, tx_pose_xy: Optional[torch.Tensor] = None) -> torch.Tensor:
        del rx_pose, tx_pose_xy
        if self.iq_receiver is None:
            self.iq_receiver = RealIQDataAcquisition(
                topic=self.real_iq_topic,
                device=self.device,
                selected_indices=self.real_iq_selected_indices,
                warmup_s=self.real_iq_warmup_s,
                timeout_s=self.real_iq_timeout_s,
                extra_frames=self.real_iq_extra_frames,
            )
        iq_processed = self.iq_receiver.collect_processed_iq(num_measurements=self.real_iq_num_measurements)
        return self.real_iq_to_observed_covariance(iq_processed)

    def measurement_summary(self, z) -> str:
        if self._is_covariance_measurement(z):
            stats = getattr(self, "_last_real_iq_stats", {}) or {}
            n_iq = int(stats.get("num_iq", 0.0))
            pwr = float(stats.get("iq_power_mean", float("nan")))
            return f"iq={n_iq} pwr={pwr:.3g}"
        try:
            return f"rays={int(self.clean_rays(z).shape[0])}"
        except Exception:
            return "measurement=?"

    def measurement_from_source(self, rx_pose: torch.Tensor, tx_pose_xy: Optional[torch.Tensor], source: Optional[str] = None) -> torch.Tensor:
        src = self._canonical_measurement_source(source or self.measurement_source)
        if src == "sionna":
            if tx_pose_xy is None:
                raise ValueError("Sionna measurement requires tx_true_xy; use measurement_source='real_iq' on the real robot")
            return self.measurement_sionna_fn(rx_pose, tx_pose_xy)
        if src == "nrp":
            if tx_pose_xy is None:
                raise ValueError("NRP self-measurement requires tx_true_xy; use measurement_source='real_iq' on the real robot")
            return self.measurement_nrp_fn(rx_pose, tx_pose_xy)
        if src == "real_iq":
            return self.measurement_real_iq_fn(rx_pose, tx_pose_xy)
        raise ValueError("measurement_source must be 'nrp', 'sionna', or 'real_iq'")

    def _load_ros_map(self, map_yaml_path: str) -> None:
        map_yaml_path = self._resolve_path(map_yaml_path)
        with open(map_yaml_path, "r", encoding="utf-8") as f:
            self.map_meta = yaml.safe_load(f)

        image_path = self.map_meta["image"]
        image_path = self._resolve_path(image_path, search_dirs=[os.path.dirname(map_yaml_path)])

        img = Image.open(image_path).convert("L")
        self.map_img = np.asarray(img, dtype=np.uint8)
        self.map_h, self.map_w = self.map_img.shape

        self.map_resolution = float(self.map_meta["resolution"])
        self.map_origin = np.asarray(self.map_meta["origin"][:2], dtype=np.float32)
        self.map_negate = int(self.map_meta.get("negate", 0))
        self.occ_thresh = float(self.map_meta.get("occupied_thresh", 0.65))
        self.free_thresh = float(self.map_meta.get("free_thresh", 0.196))

        if self.map_negate == 0:
            occ_prob = (255.0 - self.map_img.astype(np.float32)) / 255.0
        else:
            occ_prob = self.map_img.astype(np.float32) / 255.0

        self.map_occ_prob = occ_prob
        self.map_occ_mask = occ_prob > self.occ_thresh
        self.map_free_mask = occ_prob < self.free_thresh
        self.map_unknown_mask = ~(self.map_occ_mask | self.map_free_mask)

        x0, y0 = float(self.map_origin[0]), float(self.map_origin[1])
        x1 = x0 + self.map_w * self.map_resolution
        y1 = y0 + self.map_h * self.map_resolution
        self.map_extent = (x0, x1, y0, y1)

        disp = np.ones_like(self.map_occ_prob, dtype=np.float32)
        disp[self.map_unknown_mask] = 0.70
        disp[self.map_occ_mask] = 0.00
        self.map_display = np.flipud(disp)

        self._build_clearance_field()

    def _build_clearance_field(self) -> None:
        """Precompute distance-to-obstacle for robot-center clearance checks."""
        # Treat unknown cells as blocked for robot safety. Sionna wall-crossing
        # can still be selected at the RF/planning layer, but the selected Rx
        # endpoint must leave enough physical clearance for the mobile base.
        blocked = self.map_occ_mask | self.map_unknown_mask
        free_for_dist = ~blocked

        if distance_transform_edt is not None:
            self.map_clearance_m = distance_transform_edt(free_for_dist).astype(np.float32) * float(self.map_resolution)
            return

        # Dependency-free fallback: two-pass 8-neighbour chamfer distance.
        inf = np.float32(1.0e6)
        d = np.where(free_for_dist, inf, np.float32(0.0)).astype(np.float32)
        c1 = np.float32(1.0)
        c2 = np.float32(math.sqrt(2.0))
        h, w = d.shape

        for r in range(h):
            for c in range(w):
                best = d[r, c]
                if r > 0:
                    best = min(best, d[r - 1, c] + c1)
                    if c > 0:
                        best = min(best, d[r - 1, c - 1] + c2)
                    if c + 1 < w:
                        best = min(best, d[r - 1, c + 1] + c2)
                if c > 0:
                    best = min(best, d[r, c - 1] + c1)
                d[r, c] = best

        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                best = d[r, c]
                if r + 1 < h:
                    best = min(best, d[r + 1, c] + c1)
                    if c > 0:
                        best = min(best, d[r + 1, c - 1] + c2)
                    if c + 1 < w:
                        best = min(best, d[r + 1, c + 1] + c2)
                if c + 1 < w:
                    best = min(best, d[r, c + 1] + c1)
                d[r, c] = best

        self.map_clearance_m = d * float(self.map_resolution)

    def world_to_grid(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float32)
        x = xy[..., 0]
        y = xy[..., 1]
        col = np.floor((x - self.map_origin[0]) / self.map_resolution).astype(np.int64)
        grid_y = np.floor((y - self.map_origin[1]) / self.map_resolution).astype(np.int64)
        row = self.map_h - 1 - grid_y
        valid = (col >= 0) & (col < self.map_w) & (row >= 0) & (row < self.map_h)
        return row, col, valid

    def grid_to_world(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        row = np.asarray(row)
        col = np.asarray(col)
        x = self.map_origin[0] + (col.astype(np.float32) + 0.5) * self.map_resolution
        grid_y = self.map_h - 1 - row.astype(np.float32)
        y = self.map_origin[1] + (grid_y + 0.5) * self.map_resolution
        return np.stack([x, y], axis=-1)

    def invalid_zone_mask(self, xy: torch.Tensor) -> torch.Tensor:
        # For the Sionna/ROS-map test, let the ROS occupancy map define validity.
        return torch.zeros(xy.shape[:-1], dtype=torch.bool, device=xy.device)

    def within_map_bounds_mask(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0]
        y = xy[..., 1]
        x0, x1, y0, y1 = self.map_extent
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    def is_free_xy(self, xy: torch.Tensor) -> torch.Tensor:
        arr = xy.detach().cpu().numpy().reshape(-1, 2)
        row, col, valid = self.world_to_grid(arr)
        free = np.zeros(arr.shape[0], dtype=bool)
        free[valid] = self.map_free_mask[row[valid], col[valid]]
        return torch.from_numpy(free.reshape(xy.shape[:-1])).to(xy.device)

    def clearance_at_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Distance from xy to the nearest occupied/unknown cell, in metres."""
        arr = xy.detach().cpu().numpy().reshape(-1, 2)
        row, col, valid = self.world_to_grid(arr)
        clearance = np.zeros(arr.shape[0], dtype=np.float32)
        if np.any(valid):
            clearance[valid] = self.map_clearance_m[row[valid], col[valid]]
        return torch.from_numpy(clearance.reshape(xy.shape[:-1])).to(xy.device)

    def is_valid_tx_xy(self, xy: torch.Tensor) -> torch.Tensor:
        valid = self.within_map_bounds_mask(xy) & self.is_free_xy(xy) & (~self.invalid_zone_mask(xy))
        if getattr(self, "restrict_to_nrp_domain", False):
            valid = valid & self.within_nrp_domain_mask(xy)
        return valid

    def is_valid_rx_xy(self, xy: torch.Tensor) -> torch.Tensor:
        # Rx is a robot centre, not a point antenna.  Require enough centre
        # clearance for the Scout Mini's side envelope before pose/yaw-specific
        # rectangular-footprint filtering is applied.
        center_clear = self.clearance_at_xy(xy) >= float(self.robot_center_clearance_m)
        valid = self.within_map_bounds_mask(xy) & self.is_free_xy(xy) & center_clear & (~self.invalid_zone_mask(xy))
        if getattr(self, "restrict_to_nrp_domain", False):
            valid = valid & self.within_nrp_domain_mask(xy)
        return valid

    def _robot_local_footprint_points(self) -> torch.Tensor:
        """Sample the inflated rectangular Scout Mini footprint in robot frame."""
        if self._robot_local_footprint_cache is not None:
            return self._robot_local_footprint_cache

        half_l = 0.5 * float(self.robot_length_m) + float(self.robot_safety_margin_m)
        half_w = 0.5 * float(self.robot_width_m) + float(self.robot_safety_margin_m)
        ds = max(float(self.robot_footprint_sample_m), float(self.map_resolution))
        nx = max(3, int(math.ceil(2.0 * half_l / ds)) + 1)
        ny = max(3, int(math.ceil(2.0 * half_w / ds)) + 1)

        xs = torch.linspace(-half_l, half_l, steps=nx, device=self.device, dtype=torch.float32)
        ys = torch.linspace(-half_w, half_w, steps=ny, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        # Make sure the exact corners and edge midpoints are always sampled.
        extras = torch.tensor(
            [
                [-half_l, -half_w],
                [-half_l, 0.0],
                [-half_l, half_w],
                [0.0, -half_w],
                [0.0, half_w],
                [half_l, -half_w],
                [half_l, 0.0],
                [half_l, half_w],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._robot_local_footprint_cache = torch.cat([pts, extras], dim=0)
        return self._robot_local_footprint_cache

    def robot_footprint_world_points(self, pose: torch.Tensor) -> torch.Tensor:
        pose = pose.to(self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.view(1, 3)
            squeeze = True
        else:
            pose = pose.view(-1, 3)
            squeeze = False

        local = self._robot_local_footprint_points()  # [M,2]
        yaw = pose[:, 2]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        R = torch.stack(
            [
                torch.stack([c, -s], dim=-1),
                torch.stack([s, c], dim=-1),
            ],
            dim=-2,
        )  # [B,2,2]
        pts = pose[:, None, :2] + torch.einsum("bij,mj->bmi", R, local)
        return pts[0] if squeeze else pts

    def robot_footprint_corners_np(self, pose_np: np.ndarray) -> np.ndarray:
        pose_np = np.asarray(pose_np, dtype=np.float32).reshape(3)
        half_l = 0.5 * float(self.robot_length_m) + float(self.robot_safety_margin_m)
        half_w = 0.5 * float(self.robot_width_m) + float(self.robot_safety_margin_m)
        local = np.asarray(
            [[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]],
            dtype=np.float32,
        )
        c = math.cos(float(pose_np[2]))
        s = math.sin(float(pose_np[2]))
        R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return pose_np[:2][None, :] + local @ R.T

    def is_valid_rx_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Check centre clearance plus oriented rectangular footprint."""
        if pose.ndim == 1:
            pose_in = pose.view(1, 3)
            squeeze = True
        else:
            pose_in = pose.view(-1, 3)
            squeeze = False

        pose_in = pose_in.to(self.device, dtype=torch.float32)
        center_ok = self.is_valid_rx_xy(pose_in[:, :2])
        out = torch.zeros(pose_in.shape[0], device=self.device, dtype=torch.bool)
        if bool(torch.any(center_ok).item()):
            idx = torch.where(center_ok)[0]
            pts = self.robot_footprint_world_points(pose_in[idx]).reshape(-1, 2)
            fp_ok = self.within_map_bounds_mask(pts) & self.is_free_xy(pts)
            fp_ok = fp_ok.view(idx.numel(), -1).all(dim=1)
            out[idx] = fp_ok
        return out[0] if squeeze else out

    def _build_valid_point_pools(self, stride_px: int = 6) -> None:
        rows = np.arange(0, self.map_h, stride_px, dtype=np.int64)
        cols = np.arange(0, self.map_w, stride_px, dtype=np.int64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        free = self.map_free_mask[rr, cc]
        rr = rr[free]
        cc = cc[free]

        pts_world = self.grid_to_world(rr, cc)
        pts = torch.from_numpy(pts_world).to(self.device, dtype=torch.float32)
        tx_mask = self.is_valid_tx_xy(pts)
        rx_mask = self.is_valid_rx_xy(pts)
        self.valid_tx_points = pts[tx_mask]
        self.valid_rx_points = pts[rx_mask]

        if self.valid_tx_points.numel() == 0:
            raise RuntimeError("No valid TX points were found from the ROS map.")
        if self.valid_rx_points.numel() == 0:
            raise RuntimeError("No valid RX points were found from the ROS map.")

    def _bounded_valid_pool(self, role: str = "tx", bounds=None) -> torch.Tensor:
        """Select support without modifying the robot or source validity pools."""
        pool = self.valid_tx_points if role == "tx" else self.valid_rx_points
        if bounds is not None:
            mask = self.points_in_bounds(pool, bounds)
            pool = pool[mask]
        if pool.shape[0] == 0:
            raise ValueError(
                f"No valid {role.upper()} points inside requested bounds {self.format_bounds(bounds)}. "
                "Choose limits that include free map cells in the active source domain "
                "or decrease valid_pool_stride_px."
            )
        return pool

    def sample_valid_points(self, n: int, role: str = "tx", bounds=None) -> torch.Tensor:
        pool = self._bounded_valid_pool(role=role, bounds=bounds)
        idx = torch.randint(0, pool.shape[0], (int(n),), device=self.device)
        return pool[idx].clone()

    def snap_points_to_valid(self, points: torch.Tensor, role: str = "tx", bounds=None) -> torch.Tensor:
        if points.ndim == 1:
            points = points.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        points = points.to(self.device, dtype=torch.float32)
        valid_fn = self.is_valid_tx_xy if role == "tx" else self.is_valid_rx_xy
        valid_mask = valid_fn(points) & self.points_in_bounds(points, bounds)
        if torch.all(valid_mask):
            return points[0] if squeeze else points

        pool = self._bounded_valid_pool(role=role, bounds=bounds)
        fixed = points.clone()
        bad_idx = torch.where(~valid_mask)[0]
        if bad_idx.numel() > 0:
            # cdist is acceptable here because invalid repair is uncommon and the
            # valid pool is pre-strided.
            d = torch.cdist(points[bad_idx], pool)
            nn_idx = torch.argmin(d, dim=1)
            fixed[bad_idx] = pool[nn_idx]
        return fixed[0] if squeeze else fixed

    def snap_rx_pose_to_valid(self, pose: torch.Tensor, max_neighbors: int = 512) -> torch.Tensor:
        """Snap an Rx pose to the nearest pose whose centre and footprint fit."""
        pose = pose.to(self.device, dtype=torch.float32).view(3).clone()
        if bool(self.is_valid_rx_pose(pose).item()):
            return pose

        pool = self.valid_rx_points
        if pool.shape[0] == 0:
            pose[:2] = self.snap_points_to_valid(pose[:2], role="rx")
            return pose

        d = torch.norm(pool - pose[:2].view(1, 2), dim=1)
        k = min(int(max_neighbors), int(pool.shape[0]))
        order = torch.topk(d, k=k, largest=False).indices
        cand = pose.view(1, 3).repeat(k, 1)
        cand[:, :2] = pool[order]
        ok = self.is_valid_rx_pose(cand)
        if bool(torch.any(ok).item()):
            first = torch.where(ok)[0][0]
            return cand[first].detach().clone()

        # Last resort: centre-clear pose.  This should be rare and indicates the
        # configured footprint/margin is too strict for the available map cells.
        pose[:2] = self.snap_points_to_valid(pose[:2], role="rx")
        return pose

    def motion_is_collision_free(self, p_from_xy: torch.Tensor, p_to_xy: torch.Tensor, role: str = "rx") -> bool:
        p_from_xy = p_from_xy.detach().to(self.device, dtype=torch.float32).view(2)
        p_to_xy = p_to_xy.detach().to(self.device, dtype=torch.float32).view(2)
        dist = torch.norm(p_to_xy - p_from_xy).item()
        if dist < 1e-8:
            return bool((self.is_valid_rx_xy if role == "rx" else self.is_valid_tx_xy)(p_from_xy.view(1, 2))[0])
        ds = max(0.03, 2.0 * self.map_resolution)
        n = max(2, int(math.ceil(dist / ds)) + 1)
        a = torch.linspace(0.0, 1.0, n, device=self.device).view(-1, 1)
        pts = p_from_xy.view(1, 2) * (1.0 - a) + p_to_xy.view(1, 2) * a
        valid = self.is_valid_rx_xy(pts) if role == "rx" else self.is_valid_tx_xy(pts)
        return bool(torch.all(valid).item())

    def path_occupancy_stats(self, p_from_xy: torch.Tensor, p_to_xy: torch.Tensor, role: str = "rx") -> Dict[str, float]:
        """
        Measure how much map occupancy a straight Rx/Tx move crosses.

        This is intentionally a soft diagnostic, not a hard collision checker.
        In RT debug mode, an occupied segment represents a wall-penetration cost
        that can be paid when the candidate has enough RF/planning benefit.
        """
        p0 = p_from_xy.detach().to("cpu", dtype=torch.float32).view(2).numpy().astype(np.float32)
        p1 = p_to_xy.detach().to("cpu", dtype=torch.float32).view(2).numpy().astype(np.float32)
        dxy = p1 - p0
        dist = float(np.linalg.norm(dxy))

        valid_fn = self.is_valid_rx_xy if role == "rx" else self.is_valid_tx_xy
        p0_t = torch.as_tensor(p0, device=self.device, dtype=torch.float32).view(1, 2)
        p1_t = torch.as_tensor(p1, device=self.device, dtype=torch.float32).view(1, 2)
        start_free = bool(valid_fn(p0_t)[0].item())
        end_free = bool(valid_fn(p1_t)[0].item())

        if dist < 1e-9:
            center_clearance = float(self.clearance_at_xy(p0_t)[0].item())
            return {
                "dist": dist,
                "occupied_len": 0.0,
                "unknown_len": 0.0,
                "blocked_len": 0.0,
                "free_len": 0.0,
                "wall_cost": 0.0,
                "num_occupied_segments": 0.0,
                "max_contig_occupied_len": 0.0,
                "min_free_clearance": center_clearance,
                "end_clearance": center_clearance,
                "endpoint_clearance_shortfall": max(0.0, float(self.robot_center_clearance_m) - center_clearance),
                "collision_free": bool(start_free and end_free),
                "start_free": float(start_free),
                "end_free": float(end_free),
            }

        ds = max(float(self.map_resolution), 0.02)
        n = max(2, int(math.ceil(dist / ds)) + 1)
        a = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        pts = p0[None, :] * (1.0 - a) + p1[None, :] * a
        row, col, valid = self.world_to_grid(pts)

        occ = np.ones(n, dtype=bool)  # out-of-map is treated as occupied
        unk = np.zeros(n, dtype=bool)
        free = np.zeros(n, dtype=bool)
        clearance = np.zeros(n, dtype=np.float32)
        if np.any(valid):
            rv = row[valid]
            cv = col[valid]
            occ[valid] = self.map_occ_mask[rv, cv]
            unk[valid] = self.map_unknown_mask[rv, cv]
            free[valid] = self.map_free_mask[rv, cv]
            clearance[valid] = self.map_clearance_m[rv, cv]

        free_clearance = clearance[free & valid]
        if free_clearance.size > 0:
            min_free_clearance = float(np.min(free_clearance))
        else:
            min_free_clearance = 0.0
        end_clearance = float(clearance[-1])
        endpoint_clearance_shortfall = max(0.0, float(self.robot_center_clearance_m) - end_clearance)

        # Lengths are estimated on line intervals rather than point samples.
        interval_len = dist / float(max(n - 1, 1))
        occ_i = occ[:-1] | occ[1:] | (~valid[:-1]) | (~valid[1:])
        unk_i = unk[:-1] | unk[1:]
        blocked_i = occ_i | unk_i
        free_i = free[:-1] & free[1:] & (~blocked_i)

        occupied_len = float(np.count_nonzero(occ_i) * interval_len)
        unknown_len = float(np.count_nonzero(unk_i) * interval_len)
        blocked_len = float(np.count_nonzero(blocked_i) * interval_len)
        free_len = float(np.count_nonzero(free_i) * interval_len)

        # Count separate occupied penetrations and the thickest contiguous one.
        num_segments = 0
        max_run = 0
        run = 0
        prev = False
        for b in occ_i:
            if bool(b):
                run += 1
                if not prev:
                    num_segments += 1
                max_run = max(max_run, run)
            else:
                run = 0
            prev = bool(b)
        max_contig_occupied_len = float(max_run * interval_len)

        wall_cost = (
            occupied_len
            + self.wall_cross_unknown_weight * unknown_len
            + self.wall_cross_segment_penalty * float(num_segments)
        )

        return {
            "dist": dist,
            "occupied_len": occupied_len,
            "unknown_len": unknown_len,
            "blocked_len": blocked_len,
            "free_len": free_len,
            "wall_cost": float(wall_cost),
            "num_occupied_segments": float(num_segments),
            "max_contig_occupied_len": max_contig_occupied_len,
            "min_free_clearance": min_free_clearance,
            "end_clearance": end_clearance,
            "endpoint_clearance_shortfall": endpoint_clearance_shortfall,
            "collision_free": bool(np.count_nonzero(blocked_i) == 0 and start_free and end_free),
            "start_free": float(start_free),
            "end_free": float(end_free),
        }

    def path_is_admissible_for_motion(self, stats: Dict[str, float]) -> bool:
        """Return True if the path is either free or an allowed short penetration."""
        if bool(stats.get("collision_free", False)):
            return True
        if not self.allow_wall_crossing:
            return False
        if float(stats.get("end_free", 0.0)) < 0.5:
            return False
        if float(stats.get("dist", 0.0)) > self.wall_cross_max_step + 1e-6:
            return False
        if float(stats.get("occupied_len", 0.0)) > self.wall_cross_max_occupied_len:
            return False
        if float(stats.get("unknown_len", 0.0)) > self.wall_cross_max_unknown_len:
            return False
        if int(round(float(stats.get("num_occupied_segments", 0.0)))) > self.wall_cross_max_segments:
            return False
        if float(stats.get("max_contig_occupied_len", 0.0)) > self.wall_cross_max_contig_occupied_len:
            return False
        return True

    # ==================================================================
    # Sionna RT helpers
    # ==================================================================
    @torch.no_grad()
    def query_rt_rays_cross(
        self,
        rx_pose_batch: torch.Tensor,
        tx_xy_batch: torch.Tensor,
        rx_chunk_size: Optional[int] = None,
        tx_chunk_size: Optional[int] = None,
        snap_to_valid: bool = True,
    ) -> List[torch.Tensor]:
        """
        Query Sionna RT using Tester.inquire_ray cross-product semantics.

        rx_pose_batch : [R,3] or [3]
        tx_xy_batch   : [T,2] or [2]
        returns       : list length R*T, ordered by r-major then t-minor.
        """
        if self.tester is None:
            raise RuntimeError("Sionna Tester is not initialized; use measurement_source='nrp' or forward_model='nrp' without Sionna")
        if rx_pose_batch.ndim == 1:
            rx_pose_batch = rx_pose_batch.unsqueeze(0)
        if tx_xy_batch.ndim == 1:
            tx_xy_batch = tx_xy_batch.unsqueeze(0)

        rx_pose_batch = rx_pose_batch.to(self.device, dtype=torch.float32).clone()
        tx_xy_batch = tx_xy_batch.to(self.device, dtype=torch.float32).clone()
        if snap_to_valid and getattr(self, "snap_forward_rx_to_valid", False):
            rx_pose_batch[:, :2] = self.snap_points_to_valid(rx_pose_batch[:, :2], role="rx")
        if snap_to_valid:
            tx_xy_batch = self.snap_points_to_valid(tx_xy_batch, role="tx")

        R = int(rx_pose_batch.shape[0])
        T = int(tx_xy_batch.shape[0])
        out: List[Optional[torch.Tensor]] = [None for _ in range(R * T)]
        rx_chunk_size = int(rx_chunk_size or self.rt_rx_chunk_size)
        tx_chunk_size = int(tx_chunk_size or self.rt_tx_chunk_size)

        for r0 in range(0, R, rx_chunk_size):
            r1 = min(r0 + rx_chunk_size, R)
            rx_xy_chunk = rx_pose_batch[r0:r1, :2].detach().cpu().numpy().astype(np.float32)
            for t0 in range(0, T, tx_chunk_size):
                t1 = min(t0 + tx_chunk_size, T)
                tx_xy_chunk = tx_xy_batch[t0:t1].detach().cpu().numpy().astype(np.float32)
                _, batch_targets = self.tester.inquire_ray(rx_xy_chunk, tx_xy_chunk)

                idx = 0
                for rr in range(r1 - r0):
                    for tt in range(t1 - t0):
                        rays = batch_targets[idx]
                        idx += 1
                        if not torch.is_tensor(rays):
                            rays = torch.as_tensor(rays)
                        out[(r0 + rr) * T + (t0 + tt)] = rays.to(self.device, dtype=torch.float32)

        return [x if x is not None else torch.zeros(1, 6, device=self.device) for x in out]

    @torch.no_grad()
    def measurement_sionna_fn(self, rx_pose: torch.Tensor, tx_pose_xy: torch.Tensor) -> torch.Tensor:
        rays = self.query_rt_rays_cross(rx_pose.view(1, 3), tx_pose_xy.view(1, 2), rx_chunk_size=1, tx_chunk_size=1, snap_to_valid=False)
        return rays[0]

    @torch.no_grad()
    def measurement_nrp_fn(self, rx_pose: torch.Tensor, tx_pose_xy: torch.Tensor) -> torch.Tensor:
        rays = self.query_nrp_rays_cross(rx_pose.view(1, 3), tx_pose_xy.view(1, 2), snap_to_valid=False)
        return rays[0]

    @torch.no_grad()
    def measurement_fn(self, rx_pose: torch.Tensor, tx_pose_xy: Optional[torch.Tensor]) -> torch.Tensor:
        return self.measurement_from_source(rx_pose, tx_pose_xy, source=self.measurement_source)

    def pad_ray_list(self, rays_list: Sequence[torch.Tensor], max_paths: Optional[int] = None) -> torch.Tensor:
        if len(rays_list) == 0:
            return torch.zeros(0, 1, 6, device=self.device)
        max_len = max(int(r.shape[0]) for r in rays_list)
        if max_paths is not None:
            max_len = min(max_len, int(max_paths))
        max_len = max(1, max_len)
        padded = torch.zeros(len(rays_list), max_len, 6, device=self.device, dtype=torch.float32)
        for i, rays in enumerate(rays_list):
            r = rays.to(self.device, dtype=torch.float32)
            valid = self.clean_rays(r, top_k=max_len)
            n = min(valid.shape[0], max_len)
            if n > 0:
                padded[i, :n] = valid[:n]
        return padded

    @torch.no_grad()
    def covariances_from_rt_cross(
        self,
        rx_pose_batch: torch.Tensor,
        tx_xy_batch: torch.Tensor,
        rx_chunk_size: Optional[int] = None,
        tx_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if rx_pose_batch.ndim == 1:
            rx_pose_batch = rx_pose_batch.unsqueeze(0)
        if tx_xy_batch.ndim == 1:
            tx_xy_batch = tx_xy_batch.unsqueeze(0)
        rx_pose_batch = rx_pose_batch.to(self.device, dtype=torch.float32).clone()
        tx_xy_batch = tx_xy_batch.to(self.device, dtype=torch.float32).clone()
        if getattr(self, "snap_forward_rx_to_valid", False):
            rx_pose_batch[:, :2] = self.snap_points_to_valid(rx_pose_batch[:, :2], role="rx")
        tx_xy_batch = self.snap_points_to_valid(tx_xy_batch, role="tx")

        R = int(rx_pose_batch.shape[0])
        T = int(tx_xy_batch.shape[0])
        rays = self.query_rt_rays_cross(rx_pose_batch, tx_xy_batch, rx_chunk_size, tx_chunk_size)
        padded = self.pad_ray_list(rays)
        rx_rep = rx_pose_batch[:, None, :].expand(R, T, 3).reshape(R * T, 3)
        C = self.covariance_from_targets(rx_rep, padded)
        return C.reshape(R, T, self.N_R, self.N_R)

    # ==================================================================
    # RF / covariance surrogate
    # ==================================================================
    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _default_array_geometry(self) -> torch.Tensor:
        arr_cfg = self.cfg.get("array", {})
        if "positions_3d" in arr_cfg:
            U = torch.tensor(arr_cfg["positions_3d"], device=self.device, dtype=torch.float32)
            if U.ndim != 2 or U.shape[1] != 3:
                raise ValueError("array.positions_3d must have shape [N,3]")
            return U
        if "positions" in arr_cfg:
            U2 = torch.tensor(arr_cfg["positions"], device=self.device, dtype=torch.float32)
            if U2.ndim != 2 or U2.shape[1] != 2:
                raise ValueError("array.positions must have shape [N,2]")
            z = torch.zeros(U2.shape[0], 1, device=self.device, dtype=torch.float32)
            return torch.cat([U2, z], dim=-1)

        N_R = int(arr_cfg.get("num_elements", 4))
        f_c = float(self.cfg.get("rf", {}).get("f_c", 2.4e9))
        c0 = float(self.cfg.get("rf", {}).get("c", 299792458.0))
        lam = c0 / f_c
        d = float(arr_cfg.get("spacing", 0.5 * lam))
        # Center the ULA to reduce irrelevant common phase.
        x = (torch.arange(N_R, device=self.device, dtype=torch.float32) - (N_R - 1) / 2.0) * d
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1)

    def _init_rf_params(self) -> None:
        if self._rf_inited:
            return
        self.U = self._default_array_geometry()
        self.N_R = int(self.U.shape[0])
        rf_cfg = self.cfg.get("rf", {})
        self.c0 = float(rf_cfg.get("c", 299792458.0))
        self.f_c = float(rf_cfg.get("f_c", 2.4e9))
        self.lam = float(rf_cfg.get("lambda", self.c0 / self.f_c))
        self.k0 = 2.0 * math.pi / self.lam
        self.sigma_n2 = float(rf_cfg.get("sigma_n2", 1e-6))
        self.P_S = float(rf_cfg.get("P_S", 1.0))
        self._rf_inited = True

    def _angles_to_unitvec(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        if self.sionna_theta_phi:
            # Sionna RT convention: theta is zenith, phi is azimuth.
            st = torch.sin(theta)
            return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)
        # Legacy convention used by the previous draft: theta=azimuth, phi=elevation.
        cphi = torch.cos(phi)
        return torch.stack([cphi * torch.cos(theta), cphi * torch.sin(theta), torch.sin(phi)], dim=-1)

    def _denorm_alpha_linear(self, bar_alpha: torch.Tensor) -> torch.Tensor:
        # atf_testing.py stores 20*log10(|a|); converting with /10 gives |a|^2.
        alpha_db = bar_alpha * (self.P_max - self.P_min) + self.P_min
        return torch.pow(torch.tensor(10.0, device=bar_alpha.device), alpha_db / 10.0)

    def rx_gain_from_direction(self, v_local: torch.Tensor) -> torch.Tensor:
        # Receiver boresight is local +x.
        c = v_local[..., 0]
        if self.rx_pat_front_only:
            c = torch.clamp(c, min=0.0)
        else:
            c = torch.abs(c)
        g = c.pow(self.rx_pat_m)
        return torch.clamp(g, min=self.rx_pat_gmin)

    @torch.no_grad()
    def covariance_from_rays(
        self,
        rx_pose: torch.Tensor,
        alpha_bar: torch.Tensor,
        aoa_theta_g: torch.Tensor,
        aoa_phi_g: torch.Tensor,
        mask: torch.Tensor,
        jitter: float = 1e-9,
    ) -> torch.Tensor:
        self._init_rf_params()
        mask = mask.float()
        alpha_lin = self._denorm_alpha_linear(alpha_bar)

        if rx_pose.ndim == 1:
            rx_pose = rx_pose.unsqueeze(0)
        rx_pose = rx_pose.to(self.device, dtype=torch.float32)

        yaw = rx_pose[:, 2].unsqueeze(-1)
        if self.sionna_theta_phi:
            theta_local = aoa_theta_g
            phi_local = self._wrap_to_pi(aoa_phi_g - yaw)
        else:
            theta_local = self._wrap_to_pi(aoa_theta_g - yaw)
            phi_local = aoa_phi_g

        v_local = self._angles_to_unitvec(theta_local, phi_local)
        v_gain = -v_local if self.aoa_is_propagation else v_local
        v_steer = v_local if self.aoa_is_propagation else -v_local

        G_R = self.rx_gain_from_direction(v_gain)
        beta = self.P_S * alpha_lin * G_R * mask

        dot = torch.matmul(v_steer, self.U.T)
        A = torch.exp((-1j * self.k0) * dot)
        Bmat = A * torch.sqrt(torch.clamp(beta, min=0.0)).unsqueeze(-1)
        B_nrq = Bmat.transpose(1, 2)
        C = torch.matmul(B_nrq, B_nrq.conj().transpose(-1, -2))

        I = torch.eye(self.N_R, device=self.device, dtype=C.dtype).unsqueeze(0)
        C = C + (self.sigma_n2 + jitter) * I
        return self._hermitize(C)

    @torch.no_grad()
    def covariance_from_targets(self, rx_pose: torch.Tensor, target_rays: torch.Tensor) -> torch.Tensor:
        if target_rays.ndim == 2:
            target_rays = target_rays.unsqueeze(0)
        if rx_pose.ndim == 1:
            rx_pose = rx_pose.unsqueeze(0)
        target_rays = target_rays.to(self.device, dtype=torch.float32)
        rx_pose = rx_pose.to(self.device, dtype=torch.float32)

        # Dummy all-zero rows represent padding/no-path.
        valid_mask = (torch.sum(torch.abs(target_rays), dim=-1) > 1e-10).float()
        alpha_bar = target_rays[..., 0]
        aoa_theta_g = target_rays[..., 4]
        aoa_phi_g = target_rays[..., 5]
        return self.covariance_from_rays(rx_pose, alpha_bar, aoa_theta_g, aoa_phi_g, valid_mask)

    def _hermitize(self, C: torch.Tensor) -> torch.Tensor:
        return 0.5 * (C + C.conj().transpose(-1, -2))

    def _chol(self, C: torch.Tensor, eps_rel: float = 1e-6, eps_abs: float = 1e-12, max_tries: int = 5) -> torch.Tensor:
        C = self._hermitize(C)
        n = C.shape[-1]
        I = torch.eye(n, device=C.device, dtype=C.dtype)
        diag = torch.real(torch.diagonal(C, dim1=-2, dim2=-1))
        scale = diag.mean(dim=-1).clamp_min(eps_abs)
        for k in range(max_tries):
            jitter = (eps_rel * (10 ** k) * scale + eps_abs)[..., None, None]
            L, info = torch.linalg.cholesky_ex(C + jitter * I)
            if torch.all(info == 0):
                return L
        return torch.linalg.cholesky(C + (eps_rel * (10 ** max_tries) * scale + eps_abs)[..., None, None] * I)

    def _logdet_from_chol(self, L: torch.Tensor) -> torch.Tensor:
        diag = torch.real(torch.diagonal(L, dim1=-2, dim2=-1))
        return 2.0 * torch.log(diag.clamp_min(1e-30)).sum(dim=-1)

    @torch.no_grad()
    def loglik_from_observed_cov(self, C_obs: torch.Tensor, C_pred: torch.Tensor) -> torch.Tensor:
        if C_obs.ndim == 3:
            C_obs = C_obs[0]
        B = int(C_pred.shape[0])
        L = self._chol(C_pred)
        logdet = self._logdet_from_chol(L)
        C_obs_batch = C_obs.unsqueeze(0).expand(B, -1, -1)
        sol = torch.cholesky_solve(C_obs_batch, L)
        tr_term = torch.diagonal(sol.real, dim1=-2, dim2=-1).sum(dim=-1)
        return -logdet - tr_term

    # ==================================================================
    # Direct ray-feature likelihood for the Sionna-oracle PF
    # ==================================================================
    def clean_rays(self, rays: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        if rays is None:
            return torch.zeros(0, 6, device=self.device)
        rays = rays.to(self.device, dtype=torch.float32)
        if rays.ndim == 1:
            rays = rays.view(1, -1)
        if rays.shape[-1] < 6:
            raise ValueError(f"Expected ray tuples with 6 features, got shape {tuple(rays.shape)}")
        rays = rays[:, :6]
        valid = torch.sum(torch.abs(rays), dim=-1) > 1e-10
        rays = rays[valid]
        if rays.numel() == 0:
            return torch.zeros(0, 6, device=self.device)
        order = torch.argsort(rays[:, 0], descending=True)
        rays = rays[order]
        if top_k is not None:
            rays = rays[: int(top_k)]
        return rays

    def _ray_pairwise_cost(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # pred: [P,6], obs: [O,6]. Output [P,O].
        # We emphasize delay and AoA azimuth. Sionna zenith angles are usually
        # close to pi/2 in this 2D test, so they get smaller weights.
        wg = 1.25
        wt = 1.50
        waod_theta = 0.10
        waod_phi = 0.35
        waoa_theta = 0.15
        waoa_phi = 2.00

        dg = (pred[:, 0:1] - obs[:, 0].view(1, -1)).pow(2) * wg
        dtau = (pred[:, 1:2] - obs[:, 1].view(1, -1)).pow(2) * wt
        daod_theta = ((pred[:, 2:3] - obs[:, 2].view(1, -1)) / math.pi).pow(2) * waod_theta
        daod_phi = (self._wrap_to_pi(pred[:, 3:4] - obs[:, 3].view(1, -1)) / math.pi).pow(2) * waod_phi
        daoa_theta = ((pred[:, 4:5] - obs[:, 4].view(1, -1)) / math.pi).pow(2) * waoa_theta
        daoa_phi = (self._wrap_to_pi(pred[:, 5:6] - obs[:, 5].view(1, -1)) / math.pi).pow(2) * waoa_phi
        return dg + dtau + daod_theta + daod_phi + daoa_theta + daoa_phi

    def ray_loglik_single(self, observed_rays: torch.Tensor, predicted_rays: torch.Tensor) -> torch.Tensor:
        obs = self.clean_rays(observed_rays, top_k=self.ray_likelihood_top_k)
        pred = self.clean_rays(predicted_rays, top_k=self.ray_likelihood_top_k)
        n_obs = int(obs.shape[0])
        n_pred = int(pred.shape[0])

        if n_obs == 0 and n_pred == 0:
            return torch.tensor(0.0, device=self.device)
        if n_obs == 0 or n_pred == 0:
            cost = 2.0 + self.ray_count_penalty * abs(n_obs - n_pred)
            return torch.tensor(-cost / (2.0 * self.ray_likelihood_sigma ** 2), device=self.device)

        D = self._ray_pairwise_cost(pred, obs)
        obs_imp = 0.20 + obs[:, 0].clamp(0.0, 1.0)
        obs_imp = obs_imp / (obs_imp.sum() + 1e-12)
        pred_imp = 0.20 + pred[:, 0].clamp(0.0, 1.0)
        pred_imp = pred_imp / (pred_imp.sum() + 1e-12)

        obs_cost = torch.sum(obs_imp * torch.min(D, dim=0).values)
        pred_cost = torch.sum(pred_imp * torch.min(D, dim=1).values)
        count_cost = self.ray_count_penalty * abs(n_obs - n_pred) / float(max(1, self.ray_likelihood_top_k))
        rss_cost = self.ray_rss_weight * (obs[:, 0].mean() - pred[:, 0].mean()).pow(2)
        cost = obs_cost + self.ray_extra_pred_weight * pred_cost + count_cost + rss_cost
        ll = -cost / (2.0 * self.ray_likelihood_sigma ** 2)
        return torch.clamp(ll, min=-80.0, max=0.0)

    @torch.no_grad()
    def ray_loglik_batch(self, observed_rays: torch.Tensor, predicted_rays_list: Sequence[torch.Tensor]) -> torch.Tensor:
        vals = [self.ray_loglik_single(observed_rays, r) for r in predicted_rays_list]
        if len(vals) == 0:
            return torch.empty(0, device=self.device)
        return torch.stack(vals, dim=0)

    # ==================================================================
    # Belief / particle filter
    # ==================================================================
    def init_belief(
        self,
        N_p: int,
        sigma_init: float = 2.5,
        init_center: Optional[torch.Tensor] = None,
        init_bounds=None,
        init_x_range=None,
        init_y_range=None,
    ) -> Dict[str, torch.Tensor]:
        N_p = int(min(N_p, 1000))
        prior_bounds = self.combine_prior_bounds(init_bounds=init_bounds, init_x_range=init_x_range, init_y_range=init_y_range)
        self.warn_if_prior_extrapolates_nrp(prior_bounds)
        if prior_bounds is not None:
            # Validate support for Gaussian initialization too, before a long
            # localization/navigation run can begin with an unusable domain.
            self._bounded_valid_pool(role="tx", bounds=prior_bounds)

        if init_center is None:
            particles = self.sample_valid_points(N_p, role="tx", bounds=prior_bounds)
        else:
            center = init_center.to(self.device, dtype=torch.float32).view(2)
            if prior_bounds is not None and not bool(self.points_in_bounds(center.view(1, 2), prior_bounds)[0].item()):
                raise ValueError(
                    f"init_center {center.detach().cpu().tolist()} is outside init bounds {self.format_bounds(prior_bounds)}"
                )
            center = self.snap_points_to_valid(center, role="tx", bounds=prior_bounds)
            cov = (float(sigma_init) ** 2) * torch.eye(2, device=self.device)
            L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(2, device=self.device))
            eps = torch.randn(N_p, 2, device=self.device)
            particles = center.view(1, 2) + eps @ L.T
            particles = self.repair_invalid_particle_points(particles, role="tx", bounds=prior_bounds)

        w = torch.ones(N_p, device=self.device) / float(N_p)
        Sigma = torch.zeros(N_p, 2, 2, device=self.device)
        belief = {"w": w, "mu": particles, "Sigma": Sigma}
        if prior_bounds is not None:
            belief["prior_bounds"] = self._bounds_to_tensor(prior_bounds, self.device)
        return belief

    @staticmethod
    def mixture_mean_and_cov(belief: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        mu = belief["mu"]
        Sigma = belief.get("Sigma", torch.zeros(mu.shape[0], 2, 2, device=mu.device))
        mu_bar = torch.sum(w.unsqueeze(-1) * mu, dim=0)
        d = (mu - mu_bar.unsqueeze(0)).unsqueeze(-1)
        outer = d @ d.transpose(-1, -2)
        Sigma_bar = torch.sum(w.view(-1, 1, 1) * (Sigma + outer), dim=0)
        return mu_bar, Sigma_bar

    def particle_estimate(self, belief: Dict[str, torch.Tensor], top_mass: float = 0.75) -> torch.Tensor:
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        mu = belief["mu"]
        prior_bounds = self._tensor_to_bounds(belief.get("prior_bounds", None))

        # At t=0 all weights are equal.  Sorting equal weights and taking a
        # top-mass subset gives an arbitrary biased estimate, so use the full
        # weighted mean until the PF has generated a non-uniform posterior.
        if mu.shape[0] <= 1 or float((torch.max(w) - torch.min(w)).abs().item()) < 1e-8:
            est = torch.sum(w.unsqueeze(-1) * mu, dim=0)
            return self.snap_points_to_valid(est, role="tx", bounds=prior_bounds)

        order = torch.argsort(w, descending=True)
        w_sorted = w[order]
        cs = torch.cumsum(w_sorted, dim=0)
        keep_n = int(torch.searchsorted(cs, torch.tensor(float(top_mass), device=self.device)).item()) + 1
        keep_n = max(1, min(keep_n, mu.shape[0]))
        idx = order[:keep_n]
        wk = w[idx]
        wk = wk / (wk.sum() + 1e-12)
        est = torch.sum(wk.unsqueeze(-1) * mu[idx], dim=0)
        return self.snap_points_to_valid(est, role="tx", bounds=prior_bounds)

    def uncertainty_scalar(self, belief: Dict[str, torch.Tensor], top_mass: float = 0.90) -> float:
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        mu = belief["mu"]

        if mu.shape[0] <= 2 or float((torch.max(w) - torch.min(w)).abs().item()) < 1e-8:
            mk = torch.sum(w.unsqueeze(-1) * mu, dim=0)
            d = mu - mk.view(1, 2)
            cov = torch.sum(w.view(-1, 1, 1) * (d.unsqueeze(-1) @ d.unsqueeze(-2)), dim=0)
            return float(torch.sqrt(torch.trace(cov).clamp_min(0.0)).item())

        order = torch.argsort(w, descending=True)
        w_sorted = w[order]
        cs = torch.cumsum(w_sorted, dim=0)
        keep_n = int(torch.searchsorted(cs, torch.tensor(float(top_mass), device=self.device)).item()) + 1
        keep_n = max(2, min(keep_n, mu.shape[0]))
        idx = order[:keep_n]
        wk = w[idx]
        wk = wk / (wk.sum() + 1e-12)
        mk = torch.sum(wk.unsqueeze(-1) * mu[idx], dim=0)
        d = mu[idx] - mk.view(1, 2)
        cov = torch.sum(wk.view(-1, 1, 1) * (d.unsqueeze(-1) @ d.unsqueeze(-2)), dim=0)
        return float(torch.sqrt(torch.trace(cov).clamp_min(0.0)).item())

    def effective_sample_size(self, belief: Dict[str, torch.Tensor]) -> float:
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        return float((1.0 / torch.sum(w * w).clamp_min(1e-12)).item())

    def belief_support_summary(self, belief: Dict[str, torch.Tensor], tx_true_xy: Optional[torch.Tensor] = None) -> Dict[str, float]:
        mu = belief["mu"].detach()
        w = belief["w"].detach()
        w = w / (torch.sum(w) + 1e-12)
        out = {
            "n": float(mu.shape[0]),
            "x_min": float(torch.min(mu[:, 0]).item()),
            "x_max": float(torch.max(mu[:, 0]).item()),
            "y_min": float(torch.min(mu[:, 1]).item()),
            "y_max": float(torch.max(mu[:, 1]).item()),
            "x_mean": float(torch.sum(w * mu[:, 0]).item()),
            "y_mean": float(torch.sum(w * mu[:, 1]).item()),
            "ess": self.effective_sample_size(belief),
        }
        prior_bounds = self._tensor_to_bounds(belief.get("prior_bounds", None))
        out["has_prior_bounds"] = 0.0 if prior_bounds is None else 1.0
        if tx_true_xy is not None:
            tx = tx_true_xy.to(mu.device, dtype=torch.float32).view(1, 2)
            d = torch.norm(mu - tx, dim=1)
            out["nearest_true_dist"] = float(torch.min(d).item())
            out["mass_within_0p5m"] = float(torch.sum(w[d <= 0.5]).item())
            out["mass_within_1p0m"] = float(torch.sum(w[d <= 1.0]).item())
        return out

    @torch.no_grad()
    def tx_loglik_batch(
        self,
        rx_pose: torch.Tensor,
        observed_rays: torch.Tensor,
        tx_xy_batch: torch.Tensor,
        likelihood_mode: str = "hybrid",
        forward_model: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the same candidate-Tx score used by the PF update, without changing belief."""
        rx_pose = rx_pose.to(self.device, dtype=torch.float32).view(3)
        particles = self.snap_points_to_valid(tx_xy_batch, role="tx")
        N = int(particles.shape[0])
        if N == 0:
            return torch.empty(0, device=self.device)

        fwd = str(forward_model or self.forward_model).lower().strip()
        if fwd == "rt":
            fwd = "sionna"

        obs_is_cov = self._is_covariance_measurement(observed_rays)
        pred_rays: Optional[List[torch.Tensor]] = None
        mode = likelihood_mode.lower().strip()
        if mode not in {"ray", "cov", "hybrid"}:
            raise ValueError("likelihood_mode must be 'ray', 'cov', or 'hybrid'")
        if obs_is_cov:
            mode = "cov"

        ll_ray = None
        ll_cov = None
        if mode in {"ray", "hybrid"}:
            pred_rays = self.query_forward_rays_cross(
                rx_pose_batch=rx_pose.view(1, 3),
                tx_xy_batch=particles,
                forward_model=fwd,
            )
            ll_ray = self.ray_loglik_batch(observed_rays, pred_rays)

        if mode in {"cov", "hybrid"}:
            if fwd == "nrp":
                C_pred = self.covariances_from_nrp_cross(
                    rx_pose_batch=rx_pose.view(1, 3),
                    tx_xy_batch=particles,
                )[0]
            elif pred_rays is not None:
                pred_pad = self.pad_ray_list(pred_rays)
                rx_rep = rx_pose.view(1, 3).expand(N, 3)
                C_pred = self.covariance_from_targets(rx_rep, pred_pad)
            else:
                C_pred = self.covariances_from_forward_cross(
                    rx_pose_batch=rx_pose.view(1, 3),
                    tx_xy_batch=particles,
                    forward_model=fwd,
                )[0]
            if obs_is_cov:
                ll_cov = self.loglik_from_observed_iq_cov(observed_rays, C_pred)
                if bool(getattr(self, "real_iq_use_robust_likelihood", True)):
                    ll_cov = self._robust_zscore(ll_cov)
            else:
                C_obs = self.covariance_from_targets(rx_pose.view(1, 3), observed_rays)
                ll_cov = self.loglik_from_observed_cov(C_obs, C_pred)

        if mode == "ray":
            ll = ll_ray
        elif mode == "cov":
            ll = ll_cov
        else:
            assert ll_ray is not None and ll_cov is not None
            ll_ray_z = self._robust_zscore(ll_ray)
            ll_cov_z = self._robust_zscore(ll_cov)
            if fwd == "nrp":
                ll = 0.45 * ll_ray_z + 0.55 * ll_cov_z
            else:
                ll = 0.70 * ll_ray_z + 0.30 * ll_cov_z

        return torch.nan_to_num(ll, nan=-80.0, posinf=0.0, neginf=-80.0)

    @torch.no_grad()
    def truth_likelihood_diagnostics(
        self,
        rx_pose: torch.Tensor,
        observed_rays: torch.Tensor,
        belief: Dict[str, torch.Tensor],
        tx_true_xy: torch.Tensor,
        likelihood_mode: str = "hybrid",
        forward_model: Optional[str] = None,
    ) -> Dict[str, float]:
        """Post-hoc diagnostic only: ranks the known simulated Tx under the current likelihood."""
        tx_true = tx_true_xy.to(self.device, dtype=torch.float32).view(1, 2)
        particles = belief["mu"].to(self.device, dtype=torch.float32).view(-1, 2)
        q = torch.cat([tx_true, particles], dim=0)
        ll = self.tx_loglik_batch(rx_pose, observed_rays, q, likelihood_mode=likelihood_mode, forward_model=forward_model)
        if ll.numel() <= 1:
            return {}
        true_ll = ll[0]
        particle_ll = ll[1:]
        rank = int((particle_ll > true_ll).sum().item()) + 1
        percentile = 1.0 - float(rank - 1) / float(max(1, particle_ll.numel()))
        d = torch.norm(particles - tx_true, dim=1)
        w = belief["w"].to(self.device, dtype=torch.float32)
        w = w / (torch.sum(w) + 1e-12)
        return {
            "true_ll": float(true_ll.item()),
            "true_ll_rank": float(rank),
            "true_ll_percentile": float(percentile),
            "nearest_true_dist": float(torch.min(d).item()),
            "mass_within_0p5m": float(torch.sum(w[d <= 0.5]).item()),
            "mass_within_1p0m": float(torch.sum(w[d <= 1.0]).item()),
        }

    def systematic_resample(self, w: torch.Tensor) -> torch.Tensor:
        N = int(w.numel())
        w = w / (w.sum() + 1e-12)
        cdf = torch.cumsum(w, dim=0)
        cdf[-1] = 1.0
        u0 = torch.rand(1, device=w.device) / float(N)
        positions = u0 + torch.arange(N, device=w.device, dtype=torch.float32) / float(N)
        return torch.searchsorted(cdf, positions).clamp(max=N - 1)

    def repair_invalid_particle_points(self, points: torch.Tensor, role: str = "tx", max_resample_tries: int = 5, bounds=None) -> torch.Tensor:
        points = points.to(self.device, dtype=torch.float32).clone()
        valid_fn = self.is_valid_tx_xy if role == "tx" else self.is_valid_rx_xy

        def bounded_valid(z: torch.Tensor) -> torch.Tensor:
            return valid_fn(z) & self.points_in_bounds(z, bounds)

        valid = bounded_valid(points)
        for _ in range(max_resample_tries):
            if torch.all(valid):
                break
            bad = torch.where(~valid)[0]
            if bad.numel() == 0:
                break
            # First try local Gaussian nudges around invalid samples.
            points[bad] = points[bad] + 0.20 * torch.randn(bad.numel(), 2, device=self.device)
            valid = bounded_valid(points)
        if not torch.all(valid):
            bad = torch.where(~valid)[0]
            points[bad] = self.sample_valid_points(int(bad.numel()), role=role, bounds=bounds)
        return points

    def roughen_particles(
        self,
        particles: torch.Tensor,
        roughening_scale: float = 0.08,
        roughening_min_std: float = 0.03,
        roughening_max_std: float = 0.45,
        bounds=None,
    ) -> torch.Tensor:
        if roughening_scale <= 0.0:
            return particles
        std = torch.std(particles, dim=0, unbiased=False).clamp(min=roughening_min_std, max=roughening_max_std)
        noise = roughening_scale * std.view(1, 2) * torch.randn_like(particles)
        return self.repair_invalid_particle_points(particles + noise, role="tx", bounds=bounds)

    @torch.no_grad()
    def particle_filter_update(
        self,
        belief: Dict[str, torch.Tensor],
        observed_rays: torch.Tensor,
        rx_pose: torch.Tensor,
        likelihood_mode: str = "hybrid",
        forward_model: Optional[str] = None,
        resample_threshold: float = 0.50,
        roughening_scale: float = 0.10,
        global_rejuvenation_frac: float = 0.02,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Bootstrap PF update over Tx particles.

        With simulated Sionna/NRP measurements, observed_rays is a ray tuple
        tensor. With real robot BLE measurements, observed_rays is actually the
        observed complex spatial covariance [N_R,N_R] produced from IQ snapshots.
        In that case the update automatically uses a covariance/correlation
        likelihood; direct ray matching is unavailable for hardware IQ.
        """
        rx_pose = rx_pose.to(self.device, dtype=torch.float32).view(3)
        prior_bounds = self._tensor_to_bounds(belief.get("prior_bounds", None))
        particles = self.snap_points_to_valid(belief["mu"], role="tx", bounds=prior_bounds)
        w_prev = belief["w"] / (belief["w"].sum() + 1e-12)
        N = int(particles.shape[0])

        fwd = str(forward_model or self.forward_model).lower().strip()
        if fwd == "rt":
            fwd = "sionna"

        obs_is_cov = self._is_covariance_measurement(observed_rays)
        pred_rays: Optional[List[torch.Tensor]] = None

        mode = likelihood_mode.lower().strip()
        if mode not in {"ray", "cov", "hybrid"}:
            raise ValueError("likelihood_mode must be 'ray', 'cov', or 'hybrid'")
        if obs_is_cov:
            # Real IQ does not contain path tuples.  Use the NRP/Sionna predicted
            # spatial covariance and compare it against the measured covariance.
            mode = "cov"

        ll_ray = None
        ll_cov = None
        if mode in {"ray", "hybrid"}:
            pred_rays = self.query_forward_rays_cross(
                rx_pose_batch=rx_pose.view(1, 3),
                tx_xy_batch=particles,
                forward_model=fwd,
            )
            ll_ray = self.ray_loglik_batch(observed_rays, pred_rays)

        if mode in {"cov", "hybrid"}:
            if fwd == "nrp":
                C_pred = self.covariances_from_nrp_cross(
                    rx_pose_batch=rx_pose.view(1, 3),
                    tx_xy_batch=particles,
                )[0]
            elif pred_rays is not None:
                pred_pad = self.pad_ray_list(pred_rays)
                rx_rep = rx_pose.view(1, 3).expand(N, 3)
                C_pred = self.covariance_from_targets(rx_rep, pred_pad)
            else:
                C_pred = self.covariances_from_forward_cross(
                    rx_pose_batch=rx_pose.view(1, 3),
                    tx_xy_batch=particles,
                    forward_model=fwd,
                )[0]

            if obs_is_cov:
                ll_cov = self.loglik_from_observed_iq_cov(observed_rays, C_pred)
                if bool(getattr(self, "real_iq_use_robust_likelihood", True)):
                    ll_cov = self._robust_zscore(ll_cov)
            else:
                C_obs = self.covariance_from_targets(rx_pose.view(1, 3), observed_rays)
                ll_cov = self.loglik_from_observed_cov(C_obs, C_pred)

        if mode == "ray":
            ll = ll_ray
        elif mode == "cov":
            ll = ll_cov
        else:
            # Covariance and ray scores have very different numerical scales.
            assert ll_ray is not None and ll_cov is not None
            ll_ray_z = self._robust_zscore(ll_ray)
            ll_cov_z = self._robust_zscore(ll_cov)
            if fwd == "nrp":
                ll = 0.45 * ll_ray_z + 0.55 * ll_cov_z
            else:
                ll = 0.70 * ll_ray_z + 0.30 * ll_cov_z

        ll = torch.nan_to_num(ll, nan=-80.0, posinf=0.0, neginf=-80.0)
        temp = max(float(self.likelihood_temperature), 1e-6)
        logw = torch.log(w_prev + 1e-30) + ll / temp
        logw = logw - torch.max(logw)
        w_new = torch.exp(logw)
        w_sum = torch.sum(w_new)
        if not torch.isfinite(w_sum) or float(w_sum.item()) <= 0.0:
            w_new = torch.ones_like(w_prev) / float(N)
        else:
            w_new = w_new / w_sum

        ess_before = float((1.0 / torch.sum(w_new * w_new).clamp_min(1e-12)).item())
        resampled = False
        if ess_before < resample_threshold * float(N):
            idx = self.systematic_resample(w_new)
            particles = particles[idx].clone()
            particles = self.roughen_particles(particles, roughening_scale=roughening_scale, bounds=prior_bounds)
            w_new = torch.ones(N, device=self.device) / float(N)
            resampled = True

        if global_rejuvenation_frac > 0.0:
            n_global = int(round(global_rejuvenation_frac * N))
            if n_global > 0:
                replace_idx = torch.randperm(N, device=self.device)[:n_global]
                particles[replace_idx] = self.sample_valid_points(n_global, role="tx", bounds=prior_bounds)
                if resampled:
                    w_new = torch.ones(N, device=self.device) / float(N)
                else:
                    rejuvenation_mass = min(0.05, max(0.0, float(global_rejuvenation_frac)))
                    w_new = (1.0 - rejuvenation_mass) * w_new
                    w_new[replace_idx] = rejuvenation_mass / float(n_global)
                    w_new = w_new / (torch.sum(w_new) + 1e-12)

        belief_new = {
            "w": w_new,
            "mu": particles,
            "Sigma": torch.zeros(N, 2, 2, device=self.device),
        }
        if prior_bounds is not None:
            belief_new["prior_bounds"] = self._bounds_to_tensor(prior_bounds, self.device)
        stats = {
            "ess": self.effective_sample_size(belief_new),
            "ess_before_resample": ess_before,
            "resampled": float(resampled),
            "ll_min": float(torch.min(ll).item()) if ll.numel() else 0.0,
            "ll_max": float(torch.max(ll).item()) if ll.numel() else 0.0,
            "ll_std": float(torch.std(ll).item()) if ll.numel() > 1 else 0.0,
            "used_real_iq_cov": float(bool(obs_is_cov)),
        }
        if obs_is_cov:
            stats.update(getattr(self, "_last_real_iq_stats", {}) or {})
        return belief_new, stats

    def _robust_zscore(self, x: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(x)
        if not torch.any(finite):
            return torch.zeros_like(x)
        xf = x[finite]
        med_fill = torch.median(xf)
        min_fill = torch.min(xf)
        max_fill = torch.max(xf)
        x_clean = torch.where(torch.isnan(x), med_fill, x)
        x_clean = torch.where(torch.isposinf(x_clean), max_fill, x_clean)
        x_clean = torch.where(torch.isneginf(x_clean), min_fill, x_clean)
        med = torch.median(x_clean)
        mad = torch.median(torch.abs(x_clean - med)).clamp_min(1e-6)
        z = (x_clean - med) / (1.4826 * mad)
        return torch.clamp(z, -8.0, 8.0)

    # ==================================================================
    # Planning
    # ==================================================================
    def _face_heading(self, from_xy: torch.Tensor, to_xy: torch.Tensor) -> torch.Tensor:
        return torch.atan2(to_xy[..., 1] - from_xy[..., 1], to_xy[..., 0] - from_xy[..., 0])

    def generate_candidates(
        self,
        current_pose: torch.Tensor,
        estimate_xy: torch.Tensor,
        num_pos: int = 16,
        step: float = 0.75,
        num_headings: int = 5,
    ) -> torch.Tensor:
        """
        Generate footprint-valid candidate Rx poses.

        Compared with the previous Scout-footprint version, this function uses a
        wider penetration-aware candidate set:
          * multiple normal radii, not only one local step;
          * dense scan distances up to wall_cross_max_step for wall crossings;
          * several valid landing poses per bearing instead of the first one;
          * local snapping to a nearby clearance-valid landing pose when the
            ideal endpoint is just inside a wall or too close to a wall;
          * dense heading candidates for wall-crossing landings, because the
            rectangular Scout footprint may fit only when the robot is rotated
            roughly parallel to the wall or doorway.
        """
        current_pose = current_pose.to(self.device, dtype=torch.float32).view(3)
        estimate_xy = self.snap_points_to_valid(estimate_xy, role="tx")

        base_angles = torch.linspace(0.0, 2.0 * math.pi, steps=int(num_pos) + 1, device=self.device)[:-1]
        theta_est = self._face_heading(current_pose[:2].view(1, 2), estimate_xy.view(1, 2)).view(1)
        extra_offsets = torch.tensor(
            [0.0, -math.pi / 12.0, math.pi / 12.0, -math.pi / 6.0, math.pi / 6.0,
             -math.pi / 4.0, math.pi / 4.0, -math.pi / 3.0, math.pi / 3.0],
            device=self.device,
            dtype=torch.float32,
        )
        angles = torch.cat([base_angles, self._wrap_to_pi(theta_est + extra_offsets)], dim=0)
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        keep: List[torch.Tensor] = []
        keep_stats: List[Dict[str, float]] = []
        seen = set()
        key_res = max(0.05, 4.0 * float(self.map_resolution))

        def unique_heading_tensor(heads: torch.Tensor) -> torch.Tensor:
            heads = self._wrap_to_pi(heads.reshape(-1))
            vals: List[torch.Tensor] = []
            h_seen = set()
            for h in heads:
                key = int(round(float(h.item()) / 1e-3))
                if key in h_seen:
                    continue
                h_seen.add(key)
                vals.append(h.detach())
            if len(vals) == 0:
                return torch.zeros(1, device=self.device, dtype=torch.float32)
            return torch.stack(vals, dim=0)

        def headings_for_position(p_xy: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
            p_xy = p_xy.to(self.device, dtype=torch.float32).view(2)
            base_heading = self._face_heading(p_xy.view(1, 2), estimate_xy.view(1, 2)).view(1)
            if num_headings <= 1:
                face_offsets = torch.zeros(1, device=self.device, dtype=torch.float32)
            else:
                face_offsets = torch.linspace(
                    -math.pi / 2.0,
                    math.pi / 2.0,
                    steps=int(num_headings),
                    device=self.device,
                    dtype=torch.float32,
                )

            heads = [self._wrap_to_pi(base_heading + face_offsets), current_pose[2].view(1)]

            crossed = float(stats.get("blocked_len", 0.0)) > 1e-6
            if crossed:
                # For a rectangular mobile base, the heading that gives RF gain
                # and the heading that physically fits near the wall can be very
                # different.  Add global headings plus headings aligned with the
                # move vector and its lateral directions.
                n_h = max(4, int(self.wall_cross_heading_count))
                global_h = torch.linspace(0.0, 2.0 * math.pi, steps=n_h + 1, device=self.device, dtype=torch.float32)[:-1]
                move_heading = self._face_heading(current_pose[:2].view(1, 2), p_xy.view(1, 2)).view(1)
                lateral = move_heading + torch.tensor(
                    [-math.pi / 2.0, math.pi / 2.0, 0.0, math.pi],
                    device=self.device,
                    dtype=torch.float32,
                )
                heads.extend([global_h, lateral])

            return unique_heading_tensor(torch.cat(heads, dim=0))

        def position_has_valid_heading(p_xy: torch.Tensor, stats: Dict[str, float]) -> bool:
            h = headings_for_position(p_xy, stats)
            poses = torch.stack(
                [
                    p_xy[0].repeat(h.numel()),
                    p_xy[1].repeat(h.numel()),
                    h,
                ],
                dim=-1,
            )
            ok = self.is_valid_rx_pose(poses)
            return bool(torch.any(ok).item())

        def commit_position(p_xy: torch.Tensor, stats: Dict[str, float]) -> bool:
            p_xy = p_xy.to(self.device, dtype=torch.float32).view(2)
            if not bool(self.is_valid_rx_xy(p_xy.view(1, 2))[0].item()):
                return False
            if not self.path_is_admissible_for_motion(stats):
                return False
            key = (int(round(float(p_xy[0].item()) / key_res)), int(round(float(p_xy[1].item()) / key_res)))
            if key in seen:
                return False
            # Avoid accepting a landing point that later disappears when the
            # Scout footprint/yaw filter is applied.  This is the main reason
            # the previous version could fail to penetrate after adding robot
            # dimensions.
            if not position_has_valid_heading(p_xy, stats):
                return False
            seen.add(key)
            keep.append(p_xy.detach().clone())
            keep_stats.append(dict(stats))
            return True

        def snap_and_commit_nearby_landing(p_xy: torch.Tensor, stats_hint: Dict[str, float]) -> bool:
            if float(stats_hint.get("blocked_len", 0.0)) <= 1e-6:
                return False
            radius = max(0.0, float(self.wall_cross_landing_snap_radius_m))
            if radius <= 1e-9 or self.valid_rx_points.shape[0] == 0:
                return False
            d = torch.norm(self.valid_rx_points - p_xy.view(1, 2), dim=1)
            k = min(max(1, int(self.wall_cross_landing_snap_k)), int(self.valid_rx_points.shape[0]))
            idx = torch.topk(d, k=k, largest=False).indices
            added = False
            for j in idx:
                if float(d[j].item()) > radius:
                    break
                p2 = self.valid_rx_points[j]
                stats2 = self.path_occupancy_stats(current_pose[:2], p2, role="rx")
                if float(stats2.get("blocked_len", 0.0)) <= 1e-6:
                    continue
                if commit_position(p2, stats2):
                    added = True
                    break
            return added

        def add_position(p_xy: torch.Tensor, stats: Optional[Dict[str, float]] = None, allow_snap: bool = False) -> bool:
            p_xy = p_xy.to(self.device, dtype=torch.float32).view(2)
            if stats is None:
                stats = self.path_occupancy_stats(current_pose[:2], p_xy, role="rx")
            if commit_position(p_xy, stats):
                return True
            if allow_snap:
                return snap_and_commit_nearby_landing(p_xy, stats)
            return False

        # Always include rotate-in-place candidates.
        current_clear = float(self.clearance_at_xy(current_pose[:2].view(1, 2))[0].item())
        current_stats = {
            "dist": 0.0,
            "occupied_len": 0.0,
            "unknown_len": 0.0,
            "blocked_len": 0.0,
            "free_len": 0.0,
            "wall_cost": 0.0,
            "num_occupied_segments": 0.0,
            "max_contig_occupied_len": 0.0,
            "min_free_clearance": current_clear,
            "end_clearance": current_clear,
            "collision_free": True,
            "start_free": 1.0,
            "end_free": 1.0,
        }
        add_position(current_pose[:2], current_stats)

        # Multi-radius normal moves.  The footprint constraint can make one
        # short step invalid while a slightly deeper endpoint is valid.
        radii = []
        for f in self.candidate_range_factors:
            r = float(step) * float(f)
            if r <= 1e-6:
                continue
            radii.append(min(r, max(float(step), float(self.wall_cross_max_step))))
        radii = sorted(set([round(r, 4) for r in radii]))
        for r in radii:
            for i in range(dirs.shape[0]):
                p_xy = current_pose[:2] + float(r) * dirs[i]
                stats = self.path_occupancy_stats(current_pose[:2], p_xy, role="rx")
                add_position(p_xy, stats, allow_snap=False)

        # Penetration probes.  Add several free landings after the first blocked
        # segment, not just the first landing.  This matters once the robot has
        # finite dimensions: the closest landing behind the wall may not admit
        # any orientation for the Scout footprint.
        if self.allow_wall_crossing and self.wall_cross_max_step > float(step) + 1e-6:
            n_probe = max(3, int(self.wall_cross_probe_count))
            dists = torch.linspace(float(step), float(self.wall_cross_max_step), steps=n_probe, device=self.device)
            max_landings = max(1, int(self.wall_cross_max_landings_per_dir))
            for i in range(dirs.shape[0]):
                saw_blocked = False
                accepted = 0
                for d in dists[1:]:
                    p_xy = current_pose[:2] + d * dirs[i]
                    stats = self.path_occupancy_stats(current_pose[:2], p_xy, role="rx")
                    if float(stats.get("blocked_len", 0.0)) > 1e-6:
                        saw_blocked = True
                    if not saw_blocked:
                        continue
                    if float(stats.get("blocked_len", 0.0)) <= 1e-6:
                        continue
                    added = add_position(p_xy, stats, allow_snap=True)
                    if added:
                        accepted += 1
                        if accepted >= max_landings:
                            break

        if len(keep) == 0:
            keep = [current_pose[:2].detach().clone()]
            keep_stats = [current_stats]

        # Build pose candidates.  Wall-crossing positions get a denser heading
        # set so the footprint can rotate to a physically feasible orientation.
        cand: List[torch.Tensor] = []
        for p_xy, stats in zip(keep, keep_stats):
            h = headings_for_position(p_xy, stats)
            for th in h:
                cand.append(torch.stack([p_xy[0], p_xy[1], self._wrap_to_pi(th)]))

        cand_t = torch.stack(cand, dim=0)
        pose_ok = self.is_valid_rx_pose(cand_t)
        cand_t = cand_t[pose_ok]
        if cand_t.shape[0] == 0:
            return current_pose.view(1, 3).detach().clone()

        # Cheap cap before Sionna RT-MI evaluation.  Keep a generous quota of
        # wall-crossing candidates, then fill remaining slots with ordinary
        # progress/clearance candidates.  This preserves penetration options
        # without exploding the RT query count.
        max_total = int(self.max_candidate_poses)
        if max_total > 0 and cand_t.shape[0] > max_total:
            stats_list = [self.path_occupancy_stats(current_pose[:2], cand_t[k, :2], role="rx") for k in range(cand_t.shape[0])]
            blocked_len = torch.tensor([float(s.get("blocked_len", 0.0)) for s in stats_list], device=self.device, dtype=torch.float32)
            wall_cost = torch.tensor([float(s.get("wall_cost", 0.0)) for s in stats_list], device=self.device, dtype=torch.float32)
            end_clear = torch.tensor([float(s.get("end_clearance", 0.0)) for s in stats_list], device=self.device, dtype=torch.float32)
            cross_mask = blocked_len > 1e-6

            target = estimate_xy.view(1, 2)
            dist_cur = torch.norm(current_pose[:2].view(1, 2) - target).clamp_min(1e-6)
            dist_cand = torch.norm(cand_t[:, :2] - target, dim=1)
            progress = (dist_cur - dist_cand) / dist_cur
            progress_norm = self._normalize_score(progress)
            wall_cost_norm = torch.clamp(wall_cost / max(float(self.wall_cross_cost_norm), 1e-6), 0.0, 3.0)
            clearance = torch.clamp((end_clear - float(self.robot_center_clearance_m)) / max(float(self.clearance_bonus_range_m), 1e-6), 0.0, 1.0)
            bearing = self._face_heading(cand_t[:, :2], target.expand(cand_t.shape[0], 2))
            face = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(bearing - cand_t[:, 2])))
            smooth = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(cand_t[:, 2] - current_pose[2])))
            cheap_score = progress_norm + 0.20 * cross_mask.float() - 0.25 * wall_cost_norm + 0.15 * clearance + 0.05 * face + 0.05 * smooth

            keep_idx: List[torch.Tensor] = []
            current_mask = torch.norm(cand_t[:, :2] - current_pose[:2].view(1, 2), dim=1) < max(0.05, 2.0 * float(self.map_resolution))
            if bool(torch.any(current_mask).item()):
                ii = torch.where(current_mask)[0]
                n_keep = min(ii.numel(), max(1, int(num_headings)))
                local = torch.topk(cheap_score[ii], k=int(n_keep), largest=True).indices
                keep_idx.append(ii[local])

            remaining_budget = max_total - sum(int(x.numel()) for x in keep_idx)
            if remaining_budget > 0 and bool(torch.any(cross_mask).item()):
                ii = torch.where(cross_mask & (~current_mask))[0]
                if ii.numel() > 0:
                    n_cross = min(int(ii.numel()), int(self.max_wall_cross_candidates), remaining_budget)
                    local = torch.topk(cheap_score[ii], k=int(n_cross), largest=True).indices
                    keep_idx.append(ii[local])
                    remaining_budget = max_total - sum(int(x.numel()) for x in keep_idx)

            if remaining_budget > 0:
                already = torch.zeros(cand_t.shape[0], device=self.device, dtype=torch.bool)
                for part in keep_idx:
                    already[part] = True
                ii = torch.where(~already)[0]
                if ii.numel() > 0:
                    n_fill = min(int(ii.numel()), int(remaining_budget))
                    local = torch.topk(cheap_score[ii], k=int(n_fill), largest=True).indices
                    keep_idx.append(ii[local])

            if len(keep_idx) > 0:
                idx = torch.cat(keep_idx, dim=0)
                idx = torch.unique(idx, sorted=False)
                cand_t = cand_t[idx]

        # Store light diagnostics for verbose tuning.
        try:
            dbg_stats = [self.path_occupancy_stats(current_pose[:2], cand_t[k, :2], role="rx") for k in range(cand_t.shape[0])]
            n_cross = sum(1 for s in dbg_stats if float(s.get("blocked_len", 0.0)) > 1e-6)
        except Exception:
            n_cross = -1
        self._last_candidate_debug = {
            "num_positions": int(len(keep)),
            "num_candidates": int(cand_t.shape[0]),
            "num_wall_cross_candidates": int(n_cross),
            "wall_cross_max_step": float(self.wall_cross_max_step),
        }
        return cand_t

    @torch.no_grad()
    def mutual_information_candidates(
        self,
        belief: Dict[str, torch.Tensor],
        cand: torch.Tensor,
        mi_particle_limit: Optional[int] = 96,
        forward_model: Optional[str] = None,
    ) -> torch.Tensor:
        """Moment-matching covariance MI proxy, evaluated on a subset of particles."""
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        mu = self.snap_points_to_valid(belief["mu"], role="tx")
        if mi_particle_limit is not None and int(mi_particle_limit) > 0 and mu.shape[0] > int(mi_particle_limit):
            # Mix high-weight particles with random particles to preserve modes.
            k = int(mi_particle_limit)
            k_top = max(1, int(0.75 * k))
            top = torch.topk(w, k=k_top, largest=True).indices
            rest_n = k - k_top
            if rest_n > 0:
                rest = torch.multinomial(w, num_samples=rest_n, replacement=True)
                keep = torch.cat([top, rest], dim=0)
            else:
                keep = top
            mu = mu[keep]
            w = w[keep]
            w = w / (torch.sum(w) + 1e-12)
        elif mi_particle_limit is not None and int(mi_particle_limit) == 0:
            return torch.zeros(cand.shape[0], device=self.device)

        N_p = int(mu.shape[0])
        K = int(cand.shape[0])
        if N_p == 0 or K == 0:
            return torch.zeros(K, device=self.device)

        C = self.covariances_from_forward_cross(
            rx_pose_batch=cand,
            tx_xy_batch=mu,
            forward_model=forward_model or self.forward_model,
        )
        C_ki = C.reshape(K * N_p, self.N_R, self.N_R)
        L = self._chol(C_ki)
        logdet_ki = self._logdet_from_chol(L).view(K, N_p)
        term2 = torch.sum(w.view(1, N_p) * logdet_ki, dim=1)
        C_mix = torch.sum(w.view(1, N_p, 1, 1) * C, dim=1)
        Lm = self._chol(C_mix)
        logdet_mix = self._logdet_from_chol(Lm)
        mi = logdet_mix - term2
        return torch.nan_to_num(mi.real, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_score(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        xmin = torch.min(x)
        xmax = torch.max(x)
        if float((xmax - xmin).abs().item()) < 1e-9:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin + 1e-12)

    def select_next_pose(
        self,
        current_pose: torch.Tensor,
        cand: torch.Tensor,
        mi: torch.Tensor,
        estimate_xy: torch.Tensor,
        uncertainty: float,
        tx_true_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_pose = current_pose.to(self.device, dtype=torch.float32).view(3)
        estimate_xy = self.snap_points_to_valid(estimate_xy, role="tx")

        target_xy = estimate_xy
        if self.use_oracle_heading and tx_true_xy is not None:
            target_xy = tx_true_xy.to(self.device, dtype=torch.float32).view(2)

        mi_norm = self._normalize_score(mi)
        dist_cur = torch.norm(current_pose[:2] - target_xy).clamp_min(1e-6)
        dist_cand = torch.norm(cand[:, :2] - target_xy.view(1, 2), dim=1)
        progress = (dist_cur - dist_cand) / dist_cur
        progress_norm = self._normalize_score(progress)

        bearing = self._face_heading(cand[:, :2], target_xy.view(1, 2).expand(cand.shape[0], 2))
        face = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(bearing - cand[:, 2])))
        smooth = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(cand[:, 2] - current_pose[2])))

        # As uncertainty drops, increase exploitation/progress toward the target
        # estimate. The minimum keeps the robot from stalling when MI is flat.
        exploit = 1.0 / (1.0 + max(float(uncertainty), 0.0))
        base_score = (
            self.lambda_mi * (1.0 - 0.5 * exploit) * mi_norm
            + self.lambda_progress * (0.35 + 0.65 * exploit) * progress_norm
            + self.lambda_face * face
            + self.lambda_smooth * smooth
        )

        # If MI is effectively flat, ignore it and choose a progress/facing action.
        if mi.numel() > 0 and float((torch.max(mi) - torch.min(mi)).abs().item()) < self.mi_flat_threshold:
            base_score = self.lambda_progress * progress_norm + self.lambda_face * face + self.lambda_smooth * smooth

        # Soft wall-crossing model. A wall-penetrating move is admissible only if
        # the path crosses a bounded amount of occupancy and its penalized score
        # beats the best non-crossing alternative. MI comes from Sionna RT, so a
        # high-MI crossing is direct evidence that refraction/penetration is useful.
        stats_list = [self.path_occupancy_stats(current_pose[:2], cand[k, :2], role="rx") for k in range(cand.shape[0])]
        wall_cost = torch.tensor([float(s["wall_cost"]) for s in stats_list], device=self.device, dtype=torch.float32)
        blocked_len = torch.tensor([float(s["blocked_len"]) for s in stats_list], device=self.device, dtype=torch.float32)
        end_clearance = torch.tensor([float(s.get("end_clearance", 0.0)) for s in stats_list], device=self.device, dtype=torch.float32)
        cross_mask = blocked_len > 1e-6
        path_admissible = torch.tensor([self.path_is_admissible_for_motion(s) for s in stats_list], device=self.device, dtype=torch.bool)
        pose_admissible = self.is_valid_rx_pose(cand)
        admissible = path_admissible & pose_admissible

        wall_cost_norm = torch.clamp(wall_cost / max(float(self.wall_cross_cost_norm), 1e-6), 0.0, 3.0)
        clearance_margin = torch.clamp(end_clearance - float(self.robot_center_clearance_m), min=0.0)
        clearance_norm = torch.clamp(clearance_margin / max(float(self.clearance_bonus_range_m), 1e-6), 0.0, 1.0)
        rt_support = 0.70 * mi_norm + 0.30 * progress_norm
        score = (
            base_score
            - self.lambda_wall_cost * wall_cost_norm
            + self.lambda_penetration_support * cross_mask.float() * rt_support
            + self.lambda_clearance * clearance_norm
        )
        score = torch.where(admissible, score, torch.full_like(score, -1e18))

        if not bool(torch.any(admissible).item()):
            # Extremely defensive fallback; generate_candidates should always
            # include the current free pose.
            self._last_planning_debug = {"crossed_wall": False, "fallback": True}
            return current_pose.detach().clone()

        best_idx = int(torch.argmax(score).item())
        if bool(cross_mask[best_idx].item()):
            non_cross = admissible & (~cross_mask)
            if bool(torch.any(non_cross).item()):
                non_idx = torch.where(non_cross)[0]
                best_non_local = int(torch.argmax(score[non_idx]).item())
                best_non_idx = int(non_idx[best_non_local].item())
                if float(score[best_idx].item()) < float(score[best_non_idx].item()) + self.wall_cross_min_score_margin:
                    best_idx = best_non_idx

        chosen_stats = stats_list[best_idx]
        self._last_planning_debug = {
            "crossed_wall": bool(cross_mask[best_idx].item()),
            "score": float(score[best_idx].item()),
            "base_score": float(base_score[best_idx].item()),
            "mi_norm": float(mi_norm[best_idx].item()) if mi_norm.numel() else 0.0,
            "progress_norm": float(progress_norm[best_idx].item()),
            "wall_cost": float(chosen_stats["wall_cost"]),
            "occupied_len": float(chosen_stats["occupied_len"]),
            "unknown_len": float(chosen_stats["unknown_len"]),
            "num_occupied_segments": float(chosen_stats["num_occupied_segments"]),
            "max_contig_occupied_len": float(chosen_stats["max_contig_occupied_len"]),
            "end_clearance": float(chosen_stats.get("end_clearance", 0.0)),
            "min_free_clearance": float(chosen_stats.get("min_free_clearance", 0.0)),
            "clearance_norm": float(clearance_norm[best_idx].item()),
            "dist": float(chosen_stats["dist"]),
        }
        return cand[best_idx].detach().clone()


    # ==================================================================
    # Rigid antenna transform and ROS integration helpers
    # ==================================================================
    @staticmethod
    def _wrap_to_pi_float(x: float) -> float:
        return math.atan2(math.sin(float(x)), math.cos(float(x)))

    def base_to_antenna_pose(self, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert robot base_link pose(s) [x,y,yaw] to the RF antenna pose(s).

        ROS base convention: x forward, y left. The default antenna mount is on
        the right side: offset [0, -0.30] m in base_link. The antenna lobe/local
        z-axis yaw is base_yaw - pi/2, i.e. 270 deg for base yaw 0.
        """
        if not torch.is_tensor(base_pose):
            base_pose = torch.as_tensor(base_pose, dtype=torch.float32, device=self.device)
        base_pose = base_pose.to(self.device, dtype=torch.float32)
        if base_pose.ndim == 1:
            pose = base_pose.view(1, 3)
            squeeze = True
        else:
            pose = base_pose.view(-1, 3)
            squeeze = False

        yaw = pose[:, 2]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        ox = float(self.antenna_offset_x_m)
        oy = float(self.antenna_offset_y_m)
        ant_x = pose[:, 0] + c * ox - s * oy
        ant_y = pose[:, 1] + s * ox + c * oy
        ant_yaw = self._wrap_to_pi(yaw + float(self.antenna_yaw_offset_rad))
        out = torch.stack([ant_x, ant_y, ant_yaw], dim=-1)
        return out[0] if squeeze else out

    def antenna_to_base_pose(self, antenna_pose: torch.Tensor) -> torch.Tensor:
        """Inverse of base_to_antenna_pose for specifying initial goals as antenna poses."""
        if not torch.is_tensor(antenna_pose):
            antenna_pose = torch.as_tensor(antenna_pose, dtype=torch.float32, device=self.device)
        antenna_pose = antenna_pose.to(self.device, dtype=torch.float32)
        if antenna_pose.ndim == 1:
            pose = antenna_pose.view(1, 3)
            squeeze = True
        else:
            pose = antenna_pose.view(-1, 3)
            squeeze = False

        base_yaw = self._wrap_to_pi(pose[:, 2] - float(self.antenna_yaw_offset_rad))
        c = torch.cos(base_yaw)
        s = torch.sin(base_yaw)
        ox = float(self.antenna_offset_x_m)
        oy = float(self.antenna_offset_y_m)
        base_x = pose[:, 0] - (c * ox - s * oy)
        base_y = pose[:, 1] - (s * ox + c * oy)
        out = torch.stack([base_x, base_y, base_yaw], dim=-1)
        return out[0] if squeeze else out

    @staticmethod
    def _clean_frame_id(frame_id: str) -> str:
        return str(frame_id or "").strip().lstrip("/")

    def _frame_matches(self, actual: str, requested: str) -> bool:
        requested = self._clean_frame_id(requested)
        if requested == "":
            return True
        return self._clean_frame_id(actual) == requested

    def _select_transform_from_tf_message(self, msg):
        """Select the robot base transform from a tf2_msgs/TFMessage."""
        transforms = getattr(msg, "transforms", None)
        if transforms is None or len(transforms) == 0:
            raise TypeError("TFMessage contains no transforms")

        preferred_child = getattr(self, "robot_pose_child_frame", getattr(self, "gt_tf_child_frame", "base_link"))
        preferred_parent = getattr(self, "robot_pose_parent_frame", getattr(self, "gt_tf_parent_frame", ""))

        # First try the explicitly requested parent/child frame pair.
        for tr in transforms:
            child_ok = self._frame_matches(getattr(tr, "child_frame_id", ""), preferred_child)
            parent_ok = self._frame_matches(getattr(tr.header, "frame_id", ""), preferred_parent)
            if child_ok and parent_ok:
                return tr

        # Then try common ROS base frame names.
        common_children = ("base_link", "base_link", "chassis", "robot_base")
        for tr in transforms:
            if self._clean_frame_id(getattr(tr, "child_frame_id", "")) in common_children:
                return tr

        # Last fallback: if the topic carries only one transform, use it.
        if len(transforms) == 1:
            return transforms[0]

        available = ", ".join([f"{t.header.frame_id}->{t.child_frame_id}" for t in transforms[:8]])
        raise TypeError(
            "Could not select robot transform from TFMessage. "
            f"Set ~robot_pose_child_frame/~robot_pose_parent_frame. Available: {available}"
        )

    def _pose_from_msg(self, msg):
        """Extract a geometry_msgs/Pose-like object from common robot-pose messages."""
        if hasattr(msg, "transforms"):
            tr = self._select_transform_from_tf_message(msg)
            return tr.transform
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            return msg.pose.pose
        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            return msg.pose
        if hasattr(msg, "position") and hasattr(msg, "orientation"):
            return msg
        if hasattr(msg, "translation") and hasattr(msg, "rotation"):
            return msg
        raise TypeError(f"Unsupported robot pose message type: {type(msg)}")

    def _stamp_from_msg(self, msg):
        if hasattr(msg, "transforms"):
            try:
                tr = self._select_transform_from_tf_message(msg)
                if hasattr(tr, "header") and hasattr(tr.header, "stamp"):
                    return tr.header.stamp
            except Exception:
                pass
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            return msg.header.stamp
        if ROS_AVAILABLE and rospy is not None:
            return rospy.Time.now()
        return None

    def _xyyaw_from_ros_msg(self, msg) -> np.ndarray:
        pose = self._pose_from_msg(msg)

        # geometry_msgs/Pose has position/orientation; geometry_msgs/Transform has
        # translation/rotation. TFMessage topic fallbacks are also supported.
        if hasattr(pose, "position"):
            pos = pose.position
            q = pose.orientation
        elif hasattr(pose, "translation"):
            pos = pose.translation
            q = pose.rotation
        else:
            raise TypeError(f"Unsupported pose/transform object: {type(pose)}")

        quat = [q.x, q.y, q.z, q.w]
        if euler_from_quaternion is None:
            # Minimal yaw-only fallback, assuming a normalized quaternion.
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        else:
            _, _, yaw = euler_from_quaternion(quat)
        return np.asarray([pos.x, pos.y, yaw], dtype=np.float32)

    def _robot_pose_callback(self, msg) -> None:
        try:
            pose = self._xyyaw_from_ros_msg(msg)
            stamp = self._stamp_from_msg(msg)
            header = (self._select_transform_from_tf_message(msg).header
                      if hasattr(msg, "transforms") else getattr(msg, "header", None))
            source_frame = (getattr(self, "robot_pose_frame_override", "") or
                            getattr(header, "frame_id", "") or self.robot_pose_parent_frame)
            if not self._frame_matches(source_frame, self.goal_frame):
                pose_msg = self._make_pose_stamped(pose, source_frame)
                pose_msg.header.stamp = stamp
                pose = self._xyyaw_from_ros_msg(self._tf_listener.transformPose(self.goal_frame, pose_msg))
            if not np.isfinite(pose).all():
                raise ValueError("Robot pose contains nonfinite coordinates")
            self._latest_robot_pose = pose
            self._latest_robot_pose_stamp = stamp
            self._latest_robot_sample = (pose, stamp)
            # Backward-compatible aliases.
            self._latest_gt_pose = pose
            self._latest_gt_stamp = stamp
        except Exception as exc:
            if ROS_AVAILABLE and rospy is not None:
                topic = getattr(self, "robot_pose_topic", "robot pose topic")
                rospy.logwarn_throttle(2.0, f"[ATDF] Could not parse robot pose from {topic}: {exc}")

    # Backward-compatible callback name for old sim launch files.
    def _gt_pose_callback(self, msg) -> None:
        self._robot_pose_callback(msg)

    def _lookup_robot_pose_tf(self) -> Optional[torch.Tensor]:
        """Return current base pose from TF in the planning/map frame."""
        if not ROS_AVAILABLE or rospy is None or tf is None or self._tf_listener is None:
            return None
        target_frame = getattr(self, "robot_pose_parent_frame", getattr(self, "goal_frame", "map"))
        source_frame = getattr(self, "robot_pose_child_frame", "base_link")
        try:
            stamp = self._tf_listener.getLatestCommonTime(str(target_frame), str(source_frame))
            age = (rospy.Time.now() - stamp).to_sec()
            if age < 0.0 or (self.robot_pose_max_age_s > 0.0 and age > self.robot_pose_max_age_s):
                return None
            trans, rot = self._tf_listener.lookupTransform(str(target_frame), str(source_frame), stamp)
            self._measurement_stamp = stamp
            if euler_from_quaternion is None:
                x, y, z, w = rot
                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
            else:
                _, _, yaw = euler_from_quaternion(rot)
            return torch.tensor([float(trans[0]), float(trans[1]), float(yaw)], dtype=torch.float32, device=self.device)
        except Exception as exc:
            if ROS_AVAILABLE and rospy is not None:
                rospy.logwarn_throttle(
                    5.0,
                    f"[ATDF] Waiting for localization TF {target_frame}->{source_frame}: {exc}",
                )
            return None

    def _topic_pose_is_fresh(self) -> bool:
        stamp = getattr(self, "_latest_robot_pose_stamp", None)
        max_age = float(getattr(self, "robot_pose_max_age_s", 0.0))
        if stamp is None or max_age <= 0.0 or rospy is None:
            return True
        try:
            age = (rospy.Time.now() - stamp).to_sec()
            return 0.0 <= age <= max_age
        except Exception:
            return True

    def setup_ros_interfaces(
        self,
        robot_pose_topic: Optional[str] = "/amcl_pose",
        robot_pose_source: str = "auto",
        robot_pose_parent_frame: str = "map",
        robot_pose_child_frame: str = "base_link",
        goal_topic: str = "/move_base_simple/goal",
        move_base_action_name: str = "/move_base",
        use_move_base_action: bool = True,
        goal_frame: str = "map",
        action_wait_timeout: float = 20.0,
        robot_pose_max_age_s: float = 2.0,
        allow_goal_pose_fallback: bool = False,
        # Legacy simulation aliases.  Do not set these on the real robot unless
        # you intentionally want to consume a simulated robot pose topic.
        gt_pose_topic: Optional[str] = None,
        gt_tf_child_frame: Optional[str] = None,
        gt_tf_parent_frame: Optional[str] = None,
    ) -> None:
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS Python packages are not importable. Run this file inside the ROS Noetic environment.")

        self.robot_pose_source = str(robot_pose_source or "auto").lower().strip()
        if self.robot_pose_source in {"gt", "ground_truth", "groundtruth"}:
            self.robot_pose_source = "topic"
        if self.robot_pose_source not in {"auto", "tf", "topic", "none"}:
            raise ValueError("robot_pose_source must be 'auto', 'tf', 'topic', or 'none'")

        # If an old launch file still passes gt_pose_topic and no robot_pose_topic,
        # treat it as a generic pose topic.  The default real-robot path remains
        # TF map->base_link with /amcl_pose as a fallback.
        if (robot_pose_topic is None or str(robot_pose_topic).strip() == "") and gt_pose_topic:
            robot_pose_topic = gt_pose_topic

        self.robot_pose_topic = "" if robot_pose_topic is None else str(robot_pose_topic)
        self.gt_pose_topic = "" if gt_pose_topic is None else str(gt_pose_topic)
        self.goal_topic = str(goal_topic)
        self.move_base_action_name = str(move_base_action_name)
        self.use_move_base_action = bool(use_move_base_action)
        self.goal_frame = str(goal_frame)
        self.robot_pose_parent_frame = str(robot_pose_parent_frame or goal_frame or "map")
        self.robot_pose_child_frame = str(robot_pose_child_frame or "base_link")
        self.gt_tf_child_frame = str(gt_tf_child_frame or self.robot_pose_child_frame)
        self.gt_tf_parent_frame = str(gt_tf_parent_frame or self.robot_pose_parent_frame)
        self.robot_pose_max_age_s = float(robot_pose_max_age_s)
        self.allow_goal_pose_fallback = bool(allow_goal_pose_fallback)
        self.robot_pose_frame_override = str(_ros_param("~robot_pose_frame_override", ""))
        self.measurement_settle_s = float(_ros_param("~measurement_settle_s", 0.4))
        self.measurement_settle_timeout_s = float(_ros_param("~measurement_settle_timeout_s", 15.0))
        self.stopped_linear_speed = float(_ros_param("~stopped_linear_speed", 0.03))
        self.stopped_angular_speed = float(_ros_param("~stopped_angular_speed", 0.04))
        self.motion_max_age_s = float(_ros_param("~motion_max_age_s", 0.5))
        self.goal_timeout_clock = str(_ros_param("~goal_timeout_clock", "ros"))
        self.clock_stall_timeout_s = float(_ros_param("~clock_stall_timeout_s", 120.0))
        self._make_deadline(0.0)  # Validate the clock settings before navigation.
        if min(self.measurement_settle_s, self.measurement_settle_timeout_s,
               self.stopped_linear_speed, self.stopped_angular_speed, self.motion_max_age_s) <= 0:
            raise ValueError("Measurement settling durations and speed limits must be positive")
        self._motion_sub = rospy.Subscriber(str(_ros_param("~motion_odom_topic", "/odometry/filtered")),
                                            Odometry, self._motion_callback, queue_size=1)
        self._command_sub = rospy.Subscriber(str(_ros_param("~cmd_vel_topic", "/cmd_vel")),
                                             Twist, self._command_callback, queue_size=1)

        # TF is the preferred real-robot pose source.  In the uploaded navigation
        # stack, AMCL publishes map->odom and robot_localization publishes
        # odom->base_link, so lookupTransform(map, base_link) gives the pose used
        # by move_base without requiring any localization topic.
        self._tf_listener = None
        if self.robot_pose_source in {"auto", "tf", "topic"}:
            if tf is None:
                rospy.logwarn("[ATDF] python tf package unavailable; cannot use TF pose source")
            else:
                self._tf_listener = tf.TransformListener()
                rospy.sleep(0.3)

        # Optional topic fallback, usually /amcl_pose.  This can also consume
        # Odometry, PoseStamped, PoseWithCovarianceStamped, or TFMessage.
        self._robot_pose_sub = None
        real_topic = self.robot_pose_topic
        msg_class = None
        if self.robot_pose_source in {"auto", "topic"} and self.robot_pose_topic:
            if rostopic is not None:
                start = time.monotonic()
                while not rospy.is_shutdown() and msg_class is None:
                    try:
                        topic_info = rostopic.get_topic_class(self.robot_pose_topic, blocking=False)
                        if topic_info is not None:
                            msg_class, real_topic, _ = topic_info
                    except Exception:
                        msg_class = None
                    if msg_class is not None:
                        break
                    if time.monotonic() - start > 5.0:
                        break
                    rospy.sleep(0.1)
            if msg_class is None:
                rospy.logwarn(f"[ATDF] Could not introspect {self.robot_pose_topic}; falling back to PoseWithCovarianceStamped")
                msg_class = PoseWithCovarianceStamped
                real_topic = self.robot_pose_topic
            self._robot_pose_sub = rospy.Subscriber(real_topic, msg_class, self._robot_pose_callback, queue_size=10)
            # Legacy alias used by old variable names.
            self._gt_sub = self._robot_pose_sub

        self._goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self._goal_cancel_pub = rospy.Publisher(self.move_base_action_name.rstrip("/") + "/cancel", GoalID, queue_size=1)
        self._goal_marker_pub = rospy.Publisher("~next_goal", PoseStamped, queue_size=1, latch=True)
        self._estimate_pub = rospy.Publisher("~target_estimate", PoseStamped, queue_size=1, latch=True)
        self._antenna_pose_pub = rospy.Publisher("~antenna_pose", PoseStamped, queue_size=1, latch=True)
        self._particle_pub = rospy.Publisher("~particles", PoseArray, queue_size=1, latch=True)
        if bool(_ros_param("~publish_rviz_markers", True)):
            from atdf_rviz import LocalizationMarkers
            self._rviz_markers = LocalizationMarkers(
                frame_id=self.goal_frame, topic="~localization_markers",
                show_particles=bool(_ros_param("~rviz_show_particles", True)), max_particles=1000,
                show_orientations=bool(_ros_param("~rviz_show_orientations", True)),
                heading_length=float(_ros_param("~rviz_heading_length_m", 0.35)))

        self._move_base_client = None
        if self.use_move_base_action:
            self._move_base_client = actionlib.SimpleActionClient(self.move_base_action_name, MoveBaseAction)
            rospy.loginfo(f"[ATDF] Waiting for move_base action server {self.move_base_action_name}")
            if not self._move_base_client.wait_for_server(rospy.Duration(float(action_wait_timeout))):
                raise RuntimeError("move_base action server unavailable; start navigator.launch before ATDF")

        self._ros_ready = True
        topic_desc = real_topic if self._robot_pose_sub is not None else "disabled"
        tf_desc = f"{self.robot_pose_parent_frame}->{self.robot_pose_child_frame}" if self._tf_listener is not None else "disabled"
        rospy.loginfo(
            f"[ATDF] ROS interfaces ready: pose_source={self.robot_pose_source}, tf={tf_desc}, "
            f"pose_topic={topic_desc}, goal_topic={self.goal_topic}, "
            f"action={'on' if self._move_base_client is not None else 'off'}"
        )

    def _motion_callback(self, msg) -> None:
        twist = msg.twist.twist
        self._latest_motion = (msg.header.stamp, math.hypot(twist.linear.x, twist.linear.y),
                               abs(twist.angular.z))

    def _command_callback(self, msg) -> None:
        self._latest_command = (rospy.Time.now(), math.hypot(msg.linear.x, msg.linear.y),
                                abs(msg.angular.z))

    def _make_deadline(self, timeout_s):
        from atdf_timing import Deadline
        return Deadline(timeout_s, clock=getattr(self, "goal_timeout_clock", "wall"),
                        stall_timeout_s=getattr(self, "clock_stall_timeout_s", 0.0),
                        ros_now=lambda: rospy.Time.now().to_sec(), wall_now=time.monotonic)

    def _navigation_progress(self, deadline) -> str:
        parts = [f"sim={deadline.elapsed_ros_s:.1f}s wall={deadline.elapsed_wall_s:.1f}s "
                 f"real_time_factor={deadline.real_time_factor:.2f}"]
        for label, sample in (("cmd", getattr(self, "_latest_command", None)),
                              ("odom", getattr(self, "_latest_motion", None))):
            if sample is None:
                parts.append(f"{label}=unavailable")
            else:
                stamp, linear, angular = sample
                age = (rospy.Time.now() - stamp).to_sec()
                parts.append(f"{label}=({linear:.3f}m/s,{angular:.3f}rad/s,age={age:.2f}s)")
        return " ".join(parts)

    def _motion_sample_is_stopped(self, sample) -> bool:
        if sample is None:
            return False
        stamp, linear, angular = sample
        age = (rospy.Time.now() - stamp).to_sec()
        return (0.0 <= age <= self.motion_max_age_s and
                math.isfinite(linear) and math.isfinite(angular) and
                0.0 <= linear <= self.stopped_linear_speed and
                0.0 <= angular <= self.stopped_angular_speed)

    def wait_for_measurement_pose(self) -> torch.Tensor:
        """Capture a fresh pose after measured odometry stays stopped in ROS time.

        Distinct, advancing odometry stamps are required: a paused simulation or
        a stale zero-velocity sample cannot satisfy the settling interval.
        """
        deadline = self._make_deadline(self.measurement_settle_timeout_s)
        stopped_since = None
        previous_stamp = None
        while not rospy.is_shutdown() and not deadline.expired():
            sample = self._latest_motion
            if sample is not None:
                stamp = sample[0]
                valid = self._motion_sample_is_stopped(sample)
                if not valid or (previous_stamp is not None and stamp < previous_stamp):
                    stopped_since = None
                elif previous_stamp is None or stamp > previous_stamp:
                    if stopped_since is None:
                        stopped_since = stamp
                    if (stamp - stopped_since).to_sec() >= self.measurement_settle_s:
                        pose = self.wait_for_robot_pose(timeout=0.1)
                        # A recently cached pose may still predate braking.
                        # Require capture during the stopped interval and check
                        # that motion did not resume while waiting for the pose.
                        if (pose is not None and self._measurement_stamp is not None and
                                self._measurement_stamp >= stopped_since and
                                self._motion_sample_is_stopped(self._latest_motion) and
                                self._latest_motion[0] >= stamp):
                            return pose
                previous_stamp = stamp
            time.sleep(0.02)
        raise RuntimeError("Robot did not provide fresh stopped odometry and pose before RF measurement; "
                           f"{deadline.reason or 'ROS shutdown'}; {self._navigation_progress(deadline)}. "
                           "Check ~motion_odom_topic, /clock and robot pose")

    def wait_for_robot_pose(self, timeout: float = 30.0) -> Optional[torch.Tensor]:
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")
        start = time.monotonic()
        while not rospy.is_shutdown():
            if self.robot_pose_source in {"auto", "tf"}:
                pose = self._lookup_robot_pose_tf()
                if pose is not None:
                    return pose
            if self.robot_pose_source in {"auto", "topic"} and self._latest_robot_pose is not None and self._topic_pose_is_fresh():
                pose, stamp = self._latest_robot_sample
                self._measurement_stamp = stamp
                return torch.tensor(pose, dtype=torch.float32, device=self.device)
            if self.robot_pose_source == "none":
                return None
            if time.monotonic() - start > float(timeout):
                return None
            time.sleep(0.05)
        return None

    # Backward-compatible name. It now returns the localized robot pose, not ground truth.
    def wait_for_ground_truth_pose(self, timeout: float = 30.0) -> Optional[torch.Tensor]:
        return self.wait_for_robot_pose(timeout=timeout)

    def _pose_close(self, a: torch.Tensor, b: torch.Tensor, xy_tol: float, yaw_tol: float) -> bool:
        a_np = a.detach().cpu().numpy().reshape(3)
        b_np = b.detach().cpu().numpy().reshape(3)
        dxy = float(np.linalg.norm(a_np[:2] - b_np[:2]))
        dyaw = abs(self._wrap_to_pi_float(float(a_np[2] - b_np[2])))
        return dxy <= float(xy_tol) and dyaw <= float(yaw_tol)

    def _make_pose_stamped(self, pose_xyyaw, frame_id: str = "map"):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available in this Python environment")
        arr = np.asarray(pose_xyyaw, dtype=np.float64).reshape(3)
        msg = PoseStamped()
        msg.header.frame_id = str(frame_id)
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = float(arr[0])
        msg.pose.position.y = float(arr[1])
        msg.pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, float(arr[2]))
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        return msg

    def send_robot_goal_and_wait(
        self,
        robot_goal_pose: torch.Tensor,
        timeout: float = 90.0,
        xy_tolerance: float = 0.25,
        yaw_tolerance: float = 0.25,
        publish_repeats: int = 3,
    ) -> bool:
        """Send a base_link goal to move_base and wait until navigation finishes."""
        if not self._ros_ready:
            raise RuntimeError("Call setup_ros_interfaces() before sending goals")
        deadline = self._make_deadline(float(timeout))
        robot_goal_pose = robot_goal_pose.to(self.device, dtype=torch.float32).view(3)
        robot_goal_pose = self.snap_rx_pose_to_valid(robot_goal_pose)
        goal_np = robot_goal_pose.detach().cpu().numpy()
        msg = self._make_pose_stamped(goal_np, frame_id=self.goal_frame)
        self._last_goal_pose = robot_goal_pose.detach().clone()

        # RViz has its own goal topic. Sending both interfaces preempts the
        # action with duplicate simple goals and restarts local planning.
        self._goal_marker_pub.publish(msg)
        if self._move_base_client is not None:
            goal = MoveBaseGoal()
            goal.target_pose = msg
            self._move_base_client.send_goal(goal)
        else:
            self._goal_pub.publish(msg)

        next_progress_wall_s = 10.0
        while not rospy.is_shutdown() and not deadline.expired():
            if self._move_base_client is not None:
                state = self._move_base_client.get_state()
                if state == GoalStatus.SUCCEEDED:
                    # move_base checks its own localized goal tolerances. Isaac
                    # ground truth may differ slightly from AMCL; the next RF
                    # measurement always uses the actual, settled pose.
                    rospy.loginfo("[ATDF] Navigation succeeded: %s", self._navigation_progress(deadline))
                    return True
                if state in [GoalStatus.ABORTED, GoalStatus.REJECTED,
                             GoalStatus.PREEMPTED, GoalStatus.RECALLED, GoalStatus.LOST]:
                    rospy.logwarn("[ATDF] move_base ended with state=%s, detail=%s; %s", state,
                                  self._move_base_client.get_goal_status_text(), self._navigation_progress(deadline))
                    return False
            else:
                latest = self.wait_for_robot_pose(timeout=0.01)
                if latest is not None and self._pose_close(latest, robot_goal_pose, xy_tolerance, yaw_tolerance):
                    self._goal_cancel_pub.publish(GoalID())
                    return True
            if deadline.elapsed_wall_s >= next_progress_wall_s:
                rospy.loginfo("[ATDF] Navigation pending: %s", self._navigation_progress(deadline))
                next_progress_wall_s = deadline.elapsed_wall_s + 10.0
            time.sleep(0.05)

        if self._move_base_client is not None:
            self._move_base_client.cancel_goal()
        else:
            self._goal_cancel_pub.publish(GoalID())
        rospy.logwarn("[ATDF] Navigation cancelled: %s; %s", deadline.reason or "ROS shutdown",
                      self._navigation_progress(deadline))
        return False

    def publish_ros_debug(self, belief: Dict[str, torch.Tensor], estimate_xy: torch.Tensor, robot_pose: torch.Tensor, max_particles: int = 1000, measurement_stamp=None) -> None:
        if not self._ros_ready or not ROS_AVAILABLE:
            return
        now = measurement_stamp if measurement_stamp is not None else rospy.Time.now()
        if self._rviz_markers is not None:
            self._rviz_markers.record_step(
                robot_pose.detach().cpu().numpy(), estimate_xy.detach().cpu().numpy(),
                particles=belief["mu"].detach().cpu().numpy(), stamp=now,
                weights=belief["w"].detach().cpu().numpy())
        ant_pose = self.base_to_antenna_pose(robot_pose)
        ant_msg = self._make_pose_stamped(ant_pose.detach().cpu().numpy(), frame_id=self.goal_frame)
        ant_msg.header.stamp = now
        self._antenna_pose_pub.publish(ant_msg)

        est_msg = self._make_pose_stamped([float(estimate_xy[0].item()), float(estimate_xy[1].item()), 0.0], frame_id=self.goal_frame)
        est_msg.header.stamp = now
        self._estimate_pub.publish(est_msg)

        pa = PoseArray()
        pa.header.frame_id = self.goal_frame
        pa.header.stamp = now
        w = belief["w"] / (belief["w"].sum() + 1e-12)
        mu = belief["mu"]
        n = int(mu.shape[0])
        if n > int(max_particles):
            idx = torch.topk(w, k=int(max_particles), largest=True).indices
        else:
            idx = torch.arange(n, device=self.device)
        for pxy in mu[idx].detach().cpu().numpy():
            pose = Pose()
            pose.position.x = float(pxy[0])
            pose.position.y = float(pxy[1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self._particle_pub.publish(pa)

    def select_next_robot_pose(
        self,
        current_robot_pose: torch.Tensor,
        cand_robot: torch.Tensor,
        mi: torch.Tensor,
        estimate_xy: torch.Tensor,
        uncertainty: float,
        tx_true_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select the next robot base pose while scoring RF terms at antenna poses.

        Footprint/path admissibility is evaluated for the Scout base. Progress,
        MI, and facing terms use the side-mounted antenna pose and its lobe yaw.
        """
        current_robot_pose = current_robot_pose.to(self.device, dtype=torch.float32).view(3)
        cand_robot = cand_robot.to(self.device, dtype=torch.float32).view(-1, 3)
        estimate_xy = self.snap_points_to_valid(estimate_xy, role="tx")

        current_ant = self.base_to_antenna_pose(current_robot_pose)
        cand_ant = self.base_to_antenna_pose(cand_robot)

        target_xy = estimate_xy
        if self.use_oracle_heading and tx_true_xy is not None:
            target_xy = tx_true_xy.to(self.device, dtype=torch.float32).view(2)

        mi_norm = self._normalize_score(mi)
        dist_cur = torch.norm(current_ant[:2] - target_xy).clamp_min(1e-6)
        dist_cand = torch.norm(cand_ant[:, :2] - target_xy.view(1, 2), dim=1)
        progress = (dist_cur - dist_cand) / dist_cur
        progress_norm = self._normalize_score(progress)

        bearing = self._face_heading(cand_ant[:, :2], target_xy.view(1, 2).expand(cand_ant.shape[0], 2))
        face = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(bearing - cand_ant[:, 2])))
        smooth = 0.5 * (1.0 + torch.cos(self._wrap_to_pi(cand_robot[:, 2] - current_robot_pose[2])))

        exploit = 1.0 / (1.0 + max(float(uncertainty), 0.0))
        base_score = (
            self.lambda_mi * (1.0 - 0.5 * exploit) * mi_norm
            + self.lambda_progress * (0.35 + 0.65 * exploit) * progress_norm
            + self.lambda_face * face
            + self.lambda_smooth * smooth
        )
        if mi.numel() > 0 and float((torch.max(mi) - torch.min(mi)).abs().item()) < self.mi_flat_threshold:
            base_score = self.lambda_progress * progress_norm + self.lambda_face * face + self.lambda_smooth * smooth

        stats_list = [self.path_occupancy_stats(current_robot_pose[:2], cand_robot[k, :2], role="rx") for k in range(cand_robot.shape[0])]
        wall_cost = torch.tensor([float(s["wall_cost"]) for s in stats_list], device=self.device, dtype=torch.float32)
        blocked_len = torch.tensor([float(s["blocked_len"]) for s in stats_list], device=self.device, dtype=torch.float32)
        end_clearance = torch.tensor([float(s.get("end_clearance", 0.0)) for s in stats_list], device=self.device, dtype=torch.float32)
        cross_mask = blocked_len > 1e-6
        path_admissible = torch.tensor([self.path_is_admissible_for_motion(s) for s in stats_list], device=self.device, dtype=torch.bool)
        pose_admissible = self.is_valid_rx_pose(cand_robot)
        admissible = path_admissible & pose_admissible

        wall_cost_norm = torch.clamp(wall_cost / max(float(self.wall_cross_cost_norm), 1e-6), 0.0, 3.0)
        clearance_margin = torch.clamp(end_clearance - float(self.robot_center_clearance_m), min=0.0)
        clearance_norm = torch.clamp(clearance_margin / max(float(self.clearance_bonus_range_m), 1e-6), 0.0, 1.0)
        rt_support = 0.70 * mi_norm + 0.30 * progress_norm
        score = (
            base_score
            - self.lambda_wall_cost * wall_cost_norm
            + self.lambda_penetration_support * cross_mask.float() * rt_support
            + self.lambda_clearance * clearance_norm
        )
        score = torch.where(admissible, score, torch.full_like(score, -1e18))

        if not bool(torch.any(admissible).item()):
            self._last_planning_debug = {"crossed_wall": False, "fallback": True}
            return current_robot_pose.detach().clone()

        best_idx = int(torch.argmax(score).item())
        if bool(cross_mask[best_idx].item()):
            non_cross = admissible & (~cross_mask)
            if bool(torch.any(non_cross).item()):
                non_idx = torch.where(non_cross)[0]
                best_non_local = int(torch.argmax(score[non_idx]).item())
                best_non_idx = int(non_idx[best_non_local].item())
                if float(score[best_idx].item()) < float(score[best_non_idx].item()) + self.wall_cross_min_score_margin:
                    best_idx = best_non_idx

        chosen_stats = stats_list[best_idx]
        self._last_planning_debug = {
            "crossed_wall": bool(cross_mask[best_idx].item()),
            "score": float(score[best_idx].item()),
            "base_score": float(base_score[best_idx].item()),
            "mi_norm": float(mi_norm[best_idx].item()) if mi_norm.numel() else 0.0,
            "progress_norm": float(progress_norm[best_idx].item()),
            "wall_cost": float(chosen_stats["wall_cost"]),
            "occupied_len": float(chosen_stats["occupied_len"]),
            "unknown_len": float(chosen_stats["unknown_len"]),
            "num_occupied_segments": float(chosen_stats["num_occupied_segments"]),
            "max_contig_occupied_len": float(chosen_stats["max_contig_occupied_len"]),
            "end_clearance": float(chosen_stats.get("end_clearance", 0.0)),
            "min_free_clearance": float(chosen_stats.get("min_free_clearance", 0.0)),
            "clearance_norm": float(clearance_norm[best_idx].item()),
            "dist": float(chosen_stats["dist"]),
            "robot_goal_x": float(cand_robot[best_idx, 0].item()),
            "robot_goal_y": float(cand_robot[best_idx, 1].item()),
            "robot_goal_yaw": float(cand_robot[best_idx, 2].item()),
            "antenna_goal_x": float(cand_ant[best_idx, 0].item()),
            "antenna_goal_y": float(cand_ant[best_idx, 1].item()),
            "antenna_goal_yaw": float(cand_ant[best_idx, 2].item()),
        }
        return cand_robot[best_idx].detach().clone()

    @torch.no_grad()
    def run_ros_active_localization(
        self,
        tx_true_xy: Optional[torch.Tensor] = None,
        rx_init_pose: Optional[torch.Tensor] = None,
        rx_init_reference: str = "base",
        N_p: int = 1000,
        sigma_init: float = 2.5,
        T_max: int = 100,
        stop_uncertainty: float = 0.45,
        stop_rx_to_est: float = 0.60,
        success_error: float = 0.75,
        cand_num_pos: int = 16,
        cand_step: float = 0.75,
        cand_num_headings: int = 5,
        mi_particle_limit: Optional[int] = 96,
        likelihood_mode: str = "hybrid",
        planning_mode: str = "hybrid",
        forward_model: Optional[str] = None,
        measurement_source: Optional[str] = None,
        resample_threshold: float = 0.50,
        roughening_scale: float = 0.10,
        global_rejuvenation_frac: float = 0.02,
        init_bounds=None,
        init_x_range=None,
        init_y_range=None,
        debug_truth_diagnostics: bool = False,
        use_truth_success_stop: bool = False,
        truth_indicator_only: bool = True,
        goal_timeout: float = 300.0,
        goal_xy_tolerance: float = 0.25,
        goal_yaw_tolerance: float = 0.30,
        plot_every: int = 1,
        plot_dir: Optional[str] = None,
        plot_format: str = "pdf",
        show_plots: bool = False,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        ROS/Isaac Sim closed loop:
          1. initialize the Tx particle belief;
          2. send the robot base to rx_init via move_base;
          3. at arrival, read robot localization pose, transform base->antenna;
          4. collect real BLE IQ at the antenna pose (or query simulated Sionna/NRP if configured), update the PF, plan next base pose;
          5. send the next base pose and repeat.
        """
        if not self._ros_ready:
            raise RuntimeError("Call setup_ros_interfaces() before run_ros_active_localization().")

        run_settings = {name: value for name, value in locals().copy().items()
                        if name not in {"self", "tx_true_xy", "rx_init_pose"}}
        run_settings["tx_true_xy"] = (tx_true_xy.detach().cpu().numpy()
                                      if tx_true_xy is not None else None)
        run_settings["rx_init_pose"] = (rx_init_pose.detach().cpu().numpy()
                                        if rx_init_pose is not None else None)

        N_p = int(min(int(N_p), 1000))
        T_max = int(min(int(T_max), 100))
        fwd = str(forward_model or self.forward_model).lower().strip()
        if fwd == "rt":
            fwd = "sionna"
        meas_src = self._canonical_measurement_source(measurement_source or self.measurement_source)

        tx_indicator_xy = None
        if tx_true_xy is not None:
            tx_input_xy = tx_true_xy.to(self.device, dtype=torch.float32).view(2)
            # On the real robot, true Tx is only a plotting marker/indicator.
            # Do not snap it, do not use it for measurement, planning, debug likelihood, or stopping.
            tx_indicator_xy = tx_input_xy.detach().clone()
            if meas_src == "real_iq" and bool(truth_indicator_only):
                tx_true_xy = None
            else:
                # The physical source and its RViz marker must agree exactly;
                # snapping the truth would move the simulated transmitter.
                tx_true_xy = tx_input_xy.detach().clone()
        elif meas_src not in {"real_iq"}:
            raise ValueError("tx_true_xy is required for simulated Sionna/NRP measurement. Use measurement_source='real_iq' on the real robot.")

        if self._rviz_markers is not None:
            self._rviz_markers.reset(ground_truth=(tx_indicator_xy.detach().cpu().numpy()
                                                 if tx_indicator_xy is not None else None))

        if verbose:
            rospy.loginfo(f"[ATDF] ROS run: forward_model={fwd}, measurement_source={meas_src}, likelihood={likelihood_mode}")
            rospy.loginfo(
                f"[ATDF] antenna mount offset=({self.antenna_offset_x_m:+.3f},{self.antenna_offset_y_m:+.3f}) m, "
                f"yaw_offset={math.degrees(self.antenna_yaw_offset_rad):+.1f} deg"
            )
            if tx_indicator_xy is not None and meas_src == "real_iq" and bool(truth_indicator_only):
                rospy.loginfo(
                    f"[ATDF] True Tx indicator enabled for plots only: "
                    f"({tx_indicator_xy[0].item():+.3f},{tx_indicator_xy[1].item():+.3f})"
                )

        plot_format = str(plot_format or "pdf").lower().strip().lstrip(".")
        if plot_format not in {"pdf", "png", "svg"}:
            plot_format = "pdf"

        run_settings.update(N_p=N_p, T_max=T_max, forward_model=fwd,
                            measurement_source=meas_src, plot_format=plot_format,
                            init_bounds=self._parse_xy_bounds(init_bounds),
                            init_x_range=self._parse_range(init_x_range),
                            init_y_range=self._parse_range(init_y_range))

        belief = self.init_belief(
            N_p=N_p,
            sigma_init=sigma_init,
            init_bounds=init_bounds,
            init_x_range=init_x_range,
            init_y_range=init_y_range,
        )

        # Persist independently of PDF/RViz settings. Validate the destination
        # and save the prior before any navigation starts. A fresh directory
        # prevents repeated run IDs from overwriting earlier experiments.
        artifacts = RunArtifacts(
            plot_dir if plot_dir is not None else os.path.expanduser("~/.ros/atdf_video/plots"),
            self.rand_idx,
            metadata={"settings": run_settings,
                      "model_config": getattr(self, "cfg", {}),
                      "checkpoint_path": getattr(self, "checkpoint_path", None),
                      "frame": getattr(self, "goal_frame", "map"),
                      "antenna_mount": {"x_m": self.antenna_offset_x_m,
                                        "y_m": self.antenna_offset_y_m,
                                        "yaw_rad": self.antenna_yaw_offset_rad},
                      "pose_columns": ["x_m", "y_m", "yaw_rad"],
                      "measurement_source": meas_src},
        )
        plot_dir = str(artifacts.directory.resolve())
        rospy.loginfo("[ATDF] Saving run artifacts to: %s", plot_dir)
        if plot_every > 0:
            rospy.loginfo("[ATDF] Saving %s plots every %d step(s); numerical data at every measurement.",
                          plot_format.upper(), plot_every)
        else:
            rospy.logwarn("[ATDF] Plot files explicitly disabled (plot_every=%d). Numerical data still saves at every measurement in %s.",
                          plot_every, plot_dir)

        def belief_arrays():
            arrays = {"belief_" + key: belief[key].detach().cpu().numpy()
                      for key in ("mu", "w", "Sigma")}
            if "prior_bounds" in belief:
                arrays["prior_bounds"] = belief["prior_bounds"].detach().cpu().numpy()
            if tx_indicator_xy is not None:
                arrays["tx_true_xy"] = tx_indicator_xy.detach().cpu().numpy()
            return arrays

        artifacts.save_step(0, belief_arrays(), metadata={"kind": "initial_prior"})

        current_robot_pose = self.wait_for_robot_pose(timeout=30.0)
        if current_robot_pose is None:
            raise RuntimeError("No robot localization pose received from TF/topic; cannot start ATDF.")
        # Keep the localized robot pose raw for RF measurement. Planning
        # candidates are snapped/filtered separately before they are sent to ROS.
        current_robot_pose = current_robot_pose.to(self.device, dtype=torch.float32).view(3)

        if rx_init_pose is not None:
            rx_init_pose = rx_init_pose.to(self.device, dtype=torch.float32).view(3)
            if str(rx_init_reference).lower().strip() in {"antenna", "rx"}:
                init_robot_goal = self.antenna_to_base_pose(rx_init_pose)
            else:
                init_robot_goal = rx_init_pose
            init_robot_goal = self.snap_rx_pose_to_valid(init_robot_goal)
            if verbose:
                rospy.loginfo(
                    f"[ATDF] Sending initial robot goal: "
                    f"({init_robot_goal[0].item():+.3f},{init_robot_goal[1].item():+.3f},{init_robot_goal[2].item():+.3f})"
                )
            ok = self.send_robot_goal_and_wait(init_robot_goal, timeout=goal_timeout, xy_tolerance=goal_xy_tolerance, yaw_tolerance=goal_yaw_tolerance)
            if not ok:
                rospy.logwarn("[ATDF] Initial navigation did not report success; continuing from current robot localization pose.")

        latest = self.wait_for_robot_pose(timeout=5.0)
        if latest is not None:
            current_robot_pose = latest.to(self.device, dtype=torch.float32).view(3)
        current_antenna_pose = self.base_to_antenna_pose(current_robot_pose)

        traj_robot = [current_robot_pose.detach().clone()]
        traj_ant = [current_antenna_pose.detach().clone()]
        traj_est: List[torch.Tensor] = []
        traj_unc: List[torch.Tensor] = []
        traj_err: List[torch.Tensor] = []
        traj_ess: List[torch.Tensor] = []
        meas_history: List[torch.Tensor] = []
        pf_stats_history: List[Dict[str, float]] = []
        move_stats_history: List[Dict[str, float]] = []
        measurement_stamps: List[float] = []

        if plot_dir is not None and plot_every > 0:
            os.makedirs(plot_dir, exist_ok=True)
            self.plot_valid_mask(savepath=os.path.join(plot_dir, f"{self.rand_idx}_mask.{plot_format}"), show=False)
            est0 = self.particle_estimate(belief)
            self.plot_state(
                belief=belief,
                rx_traj=torch.stack(traj_ant, dim=0),
                robot_traj=torch.stack(traj_robot, dim=0),
                estimate_xy=est0,
                tx_true_xy=tx_indicator_xy,
                title=f"ATDF-{fwd.upper()} ROS step 0 (initial belief)",
                savepath=os.path.join(plot_dir, f"{self.rand_idx}_000.{plot_format}"),
                show=show_plots,
            )

        found = False
        success = False
        for t in range(1, T_max + 1):
            if rospy.is_shutdown():
                break

            current_robot_pose = self.wait_for_measurement_pose().to(self.device, dtype=torch.float32).view(3)
            measurement_stamp = self._measurement_stamp
            current_antenna_pose = self.base_to_antenna_pose(current_robot_pose)
            traj_robot[-1] = current_robot_pose.detach().clone()
            traj_ant[-1] = current_antenna_pose.detach().clone()

            observed_rays = self.measurement_from_source(current_antenna_pose, tx_true_xy, source=meas_src)
            meas_history.append(observed_rays.detach().clone())

            belief, pf_stats = self.particle_filter_update(
                belief=belief,
                observed_rays=observed_rays,
                rx_pose=current_antenna_pose,
                likelihood_mode=likelihood_mode,
                forward_model=fwd,
                resample_threshold=resample_threshold,
                roughening_scale=roughening_scale,
                global_rejuvenation_frac=global_rejuvenation_frac,
            )
            if debug_truth_diagnostics and tx_true_xy is not None:
                truth_diag = self.truth_likelihood_diagnostics(
                    rx_pose=current_antenna_pose,
                    observed_rays=observed_rays,
                    belief=belief,
                    tx_true_xy=tx_true_xy,
                    likelihood_mode=likelihood_mode,
                    forward_model=fwd,
                )
                pf_stats.update(truth_diag)
            pf_stats_history.append(pf_stats)

            est_xy = self.particle_estimate(belief)
            unc = self.uncertainty_scalar(belief)
            err = float(torch.norm(est_xy - tx_true_xy).item()) if tx_true_xy is not None else float("nan")
            ant_to_est = float(torch.norm(current_antenna_pose[:2] - est_xy).item())
            ess = self.effective_sample_size(belief)

            traj_est.append(est_xy.detach().clone())
            traj_unc.append(torch.tensor(unc, device=self.device))
            traj_err.append(torch.tensor(err, device=self.device))
            traj_ess.append(torch.tensor(ess, device=self.device))
            measurement_stamps.append(measurement_stamp.to_sec() if measurement_stamp is not None else float("nan"))
            arrays = belief_arrays()
            arrays["measurement"] = observed_rays.detach().cpu().numpy()
            for name, history in (("robot_traj", traj_robot), ("antenna_traj", traj_ant),
                                  ("est_traj", traj_est), ("unc_traj", traj_unc),
                                  ("err_traj", traj_err), ("ess_traj", traj_ess)):
                arrays[name] = torch.stack(history, dim=0).detach().cpu().numpy()
            arrays["measurement_stamps"] = np.asarray(measurement_stamps, dtype=np.float64)
            # Save before plotting or publishing: their failure must not lose
            # this measurement and its complete posterior distribution.
            artifacts.save_step(t, arrays, metadata={"kind": "measurement_posterior",
                                                      "pf_stats": pf_stats,
                                                      "move_stats": move_stats_history,
                                                      "antenna_to_estimate_m": ant_to_est})
            self.publish_ros_debug(belief, est_xy, current_robot_pose, measurement_stamp=measurement_stamp)

            if verbose:
                msg = (
                    f"[t={t:03d}] robot=({current_robot_pose[0].item():+.3f},{current_robot_pose[1].item():+.3f},{current_robot_pose[2].item():+.3f}) "
                    f"antenna=({current_antenna_pose[0].item():+.3f},{current_antenna_pose[1].item():+.3f},{current_antenna_pose[2].item():+.3f}) "
                    f"est=({est_xy[0].item():+.3f},{est_xy[1].item():+.3f}) "
                    f"err={err:.3f} unc={unc:.3f} ant_to_est={ant_to_est:.3f} "
                    f"ESS={ess:.1f}/{N_p} {self.measurement_summary(observed_rays)} "
                    f"ll=[{pf_stats['ll_min']:.2f},{pf_stats['ll_max']:.2f}]"
                )
                if debug_truth_diagnostics and "true_ll_rank" in pf_stats:
                    msg += (
                        f" true_rank={int(pf_stats['true_ll_rank'])}/{N_p + 1} "
                        f"true_pct={pf_stats['true_ll_percentile']:.2f} "
                        f"mass1m={pf_stats['mass_within_1p0m']:.3f}"
                    )
                rospy.loginfo(msg)

            if plot_dir is not None and plot_every > 0 and (t % plot_every == 0 or t == 1):
                self.plot_state(
                    belief=belief,
                    rx_traj=torch.stack(traj_ant, dim=0),
                    robot_traj=torch.stack(traj_robot, dim=0),
                    estimate_traj=torch.stack(traj_est, dim=0),
                    estimate_xy=est_xy,
                    tx_true_xy=tx_indicator_xy,
                    title=f"ATDF-{fwd.upper()} ROS step {t}",
                    savepath=os.path.join(plot_dir, f"{self.rand_idx}_{t:03d}.{plot_format}"),
                    show=show_plots,
                )

            success = (err <= success_error) if tx_true_xy is not None else False
            found_by_belief = (unc <= stop_uncertainty and ant_to_est <= stop_rx_to_est)
            found = found_by_belief or (bool(use_truth_success_stop) and success)
            if found:
                reason = "truth-success" if (success and bool(use_truth_success_stop) and not found_by_belief) else "uncertainty-and-range"
                rospy.loginfo(f"[ATDF] Stop at step {t}: {reason}; err={err:.3f}, unc={unc:.3f}, ant_to_est={ant_to_est:.3f}")
                break

            if t == T_max:
                break  # No unmeasured navigation leg after the final PF update.

            plan_robot_pose = self.snap_rx_pose_to_valid(current_robot_pose)
            cand_robot = self.generate_candidates(
                current_pose=plan_robot_pose,
                estimate_xy=est_xy,
                num_pos=cand_num_pos,
                step=cand_step,
                num_headings=cand_num_headings,
            )
            cdbg = getattr(self, "_last_candidate_debug", {})
            if verbose and cdbg:
                rospy.loginfo(
                    "[ATDF] candidates: "
                    f"K={int(cdbg.get('num_candidates', cand_robot.shape[0]))}, "
                    f"positions={int(cdbg.get('num_positions', 0))}, "
                    f"wallK={int(cdbg.get('num_wall_cross_candidates', 0))}, "
                    f"wall_step={float(cdbg.get('wall_cross_max_step', self.wall_cross_max_step)):.2f}m"
                )

            cand_ant = self.base_to_antenna_pose(cand_robot)
            if planning_mode.lower().strip() in {"mi", "hybrid"}:
                mi = self.mutual_information_candidates(belief, cand_ant, mi_particle_limit=mi_particle_limit, forward_model=fwd)
            else:
                mi = torch.zeros(cand_robot.shape[0], device=self.device)

            next_robot_pose = self.select_next_robot_pose(
                current_robot_pose=plan_robot_pose,
                cand_robot=cand_robot,
                mi=mi,
                estimate_xy=est_xy,
                uncertainty=unc,
                tx_true_xy=tx_true_xy,
            )
            move_stats = self.path_occupancy_stats(plan_robot_pose[:2], next_robot_pose[:2], role="rx")
            move_stats_history.append(move_stats)
            if verbose:
                dbg = getattr(self, "_last_planning_debug", {})
                rospy.loginfo(
                    f"[ATDF] next robot goal=({next_robot_pose[0].item():+.3f},{next_robot_pose[1].item():+.3f},{next_robot_pose[2].item():+.3f}), "
                    f"antenna goal=({dbg.get('antenna_goal_x', float('nan')):+.3f},{dbg.get('antenna_goal_y', float('nan')):+.3f},{dbg.get('antenna_goal_yaw', float('nan')):+.3f}), "
                    f"score={float(dbg.get('score', 0.0)):.3f}, mi_norm={float(dbg.get('mi_norm', 0.0)):.3f}, "
                    f"wall_cost={float(dbg.get('wall_cost', 0.0)):.2f}"
                )

            nav_ok = self.send_robot_goal_and_wait(next_robot_pose, timeout=goal_timeout, xy_tolerance=goal_xy_tolerance, yaw_tolerance=goal_yaw_tolerance)
            if not nav_ok:
                rospy.logwarn("[ATDF] Navigation did not report success; measuring from the current robot localization pose and replanning.")

            latest = self.wait_for_robot_pose(timeout=5.0)
            if latest is not None:
                current_robot_pose = latest.to(self.device, dtype=torch.float32).view(3)
            elif bool(getattr(self, "allow_goal_pose_fallback", False)):
                rospy.logwarn("[ATDF] Robot pose unavailable after navigation; using commanded goal because allow_goal_pose_fallback=true.")
                current_robot_pose = self.snap_rx_pose_to_valid(next_robot_pose)
            else:
                raise RuntimeError("Robot pose estimate unavailable after navigation; check TF/localization before collecting RF measurement.")
            current_antenna_pose = self.base_to_antenna_pose(current_robot_pose)
            traj_robot.append(current_robot_pose.detach().clone())
            traj_ant.append(current_antenna_pose.detach().clone())

        est_final = self.particle_estimate(belief)
        _, cov_final = self.mixture_mean_and_cov(belief)
        return {
            "artifact_dir": plot_dir,
            "belief_w": belief["w"].detach(),
            "belief_mu": belief["mu"].detach(),
            "belief_Sigma": belief["Sigma"].detach(),
            "robot_traj": torch.stack(traj_robot, dim=0),
            "antenna_traj": torch.stack(traj_ant, dim=0),
            "rx_traj": torch.stack(traj_ant, dim=0),
            "est_traj": torch.stack(traj_est, dim=0) if len(traj_est) > 0 else torch.empty(0, 2, device=self.device),
            "unc_traj": torch.stack(traj_unc, dim=0) if len(traj_unc) > 0 else torch.empty(0, device=self.device),
            "err_traj": torch.stack(traj_err, dim=0) if len(traj_err) > 0 else torch.empty(0, device=self.device),
            "ess_traj": torch.stack(traj_ess, dim=0) if len(traj_ess) > 0 else torch.empty(0, device=self.device),
            "measurement_rays": meas_history,
            "pf_stats": pf_stats_history,
            "move_stats": move_stats_history,
            "est_final": est_final.detach(),
            "cov_final": cov_final.detach(),
            "found": torch.tensor(bool(found), device=self.device),
            "success": torch.tensor(bool(success), device=self.device),
        }

    # ==================================================================
    # Plotting
    # ==================================================================
    def plot_valid_mask(self, savepath: Optional[str] = None, show: bool = True, max_points: int = 30000) -> None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(self.map_display, extent=self.map_extent, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        tx_pts = self.valid_tx_points.detach().cpu().numpy()
        rx_pts = self.valid_rx_points.detach().cpu().numpy()
        if tx_pts.shape[0] > max_points:
            tx_pts = tx_pts[np.random.choice(tx_pts.shape[0], size=max_points, replace=False)]
        if rx_pts.shape[0] > max_points:
            rx_pts = rx_pts[np.random.choice(rx_pts.shape[0], size=max_points, replace=False)]
        if tx_pts.shape[0] > 0:
            ax.scatter(tx_pts[:, 0], tx_pts[:, 1], s=3, c="tab:red", alpha=0.25, label="Valid TX pool")
        if rx_pts.shape[0] > 0:
            ax.scatter(rx_pts[:, 0], rx_pts[:, 1], s=3, c="tab:blue", alpha=0.25, label="Valid RX pool")
        ax.set_xlim(self.map_extent[0], self.map_extent[1])
        ax.set_ylim(self.map_extent[2], self.map_extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Valid free-space mask")
        ax.legend(loc="upper right")
        plt.tight_layout()
        if savepath is not None:
            out_dir = os.path.dirname(savepath)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        if show and matplotlib.get_backend().lower() != "agg":
            try:
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass
        plt.close(fig)

    def plot_state(
        self,
        belief: Dict[str, torch.Tensor],
        rx_traj: torch.Tensor,
        estimate_xy: Optional[torch.Tensor] = None,
        tx_true_xy: Optional[torch.Tensor] = None,
        title: str = "ATDF-Sionna target finding",
        savepath: Optional[str] = None,
        show: bool = True,
        max_particles: int = 1000,
        particle_alpha: float = 0.35,
        particle_size: float = 12.0,
        weight_size_scale: float = 220.0,
        draw_uncertainty_ellipse: bool = True,
        robot_traj: Optional[torch.Tensor] = None,
        estimate_traj: Optional[torch.Tensor] = None,
    ) -> None:
        """Save the paper-style map, measured poses, and estimate history.

        ROS callers supply base poses explicitly; ``rx_traj`` remains a
        compatible fallback for offline callers. All stored coordinates stay
        in map x/y order; the renderer handles the reference figure's axes.
        """
        def as_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        estimates = as_numpy(estimate_traj)
        if estimates is None and estimate_xy is not None:
            estimates = as_numpy(estimate_xy).reshape(1, 2)
        fig = make_reference_plot(
            map_display=self.map_display,
            map_extent=self.map_extent,
            robot_traj=as_numpy(robot_traj if robot_traj is not None else rx_traj),
            estimate_traj=estimates,
            ground_truth=as_numpy(tx_true_xy),
            particles=as_numpy(belief["mu"]),
            weights=as_numpy(belief["w"]),
            covariances=as_numpy(belief.get("Sigma")),
            roi_bounds=as_numpy(belief.get("prior_bounds")),
            title=title,
            max_particles=max_particles,
            particle_alpha=particle_alpha,
            particle_size=particle_size,
            weight_size_scale=weight_size_scale,
            draw_covariances=draw_uncertainty_ellipse,
        )
        try:
            if savepath is not None:
                out_dir = os.path.dirname(savepath)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                    fig.savefig(savepath, dpi=200, bbox_inches="tight", metadata={"Title": str(title)})
            if show and matplotlib.get_backend().lower() != "agg":
                try:
                    plt.show(block=False)
                    plt.pause(0.001)
                except Exception:
                    pass
        finally:
            plt.close(fig)

    # ==================================================================
    # Main loop
    # ==================================================================
    @torch.no_grad()
    def run_active_localization(
        self,
        tx_true_xy: torch.Tensor,
        rx_init_pose: torch.Tensor,
        measurement_fn=None,
        N_p: int = 1000,
        sigma_init: float = 2.5,
        J: int = 1,  # compatibility only; bootstrap PF does not use J
        T_max: int = 100,
        stop_uncertainty: float = 0.45,
        stop_rx_to_est: float = 0.60,
        success_error: float = 0.75,
        cand_num_pos: int = 16,
        cand_step: float = 0.75,
        cand_num_headings: int = 5,
        mi_particle_limit: Optional[int] = 96,
        likelihood_mode: str = "hybrid",
        planning_mode: str = "hybrid",
        forward_model: Optional[str] = None,
        measurement_source: Optional[str] = None,
        resample_threshold: float = 0.50,
        roughening_scale: float = 0.10,
        global_rejuvenation_frac: float = 0.02,
        init_bounds=None,
        init_x_range=None,
        init_y_range=None,
        debug_truth_diagnostics: bool = False,
        use_truth_success_stop: bool = False,
        plot_every: int = 5,
        plot_dir: Optional[str] = None,
        plot_format: str = "pdf",
        show_plots: bool = False,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Closed-loop target finding.

        Limits requested for this debug stage are enforced internally:
            T_max <= 100, N_p <= 1000
        """
        del J
        N_p = int(min(int(N_p), 1000))
        T_max = int(min(int(T_max), 100))

        tx_true_xy = tx_true_xy.to(self.device, dtype=torch.float32).view(2)
        rx_pose = rx_init_pose.to(self.device, dtype=torch.float32).view(3)
        tx_true_xy = self.snap_points_to_valid(tx_true_xy, role="tx")
        rx_pose = self.snap_rx_pose_to_valid(rx_pose)

        fwd = str(forward_model or self.forward_model).lower().strip()
        if fwd == "rt":
            fwd = "sionna"
        meas_src = str(measurement_source or self.measurement_source).lower().strip()
        if meas_src == "rt":
            meas_src = "sionna"
        if measurement_fn is None:
            measurement_fn = lambda rx, tx: self.measurement_from_source(rx, tx, source=meas_src)

        if verbose:
            print(f"[ATDF] forward_model={fwd}, measurement_source={meas_src}, likelihood={likelihood_mode}")
            print(f"[ATDF] use_oracle_heading={self.use_oracle_heading}, use_truth_success_stop={use_truth_success_stop}")

        plot_format = str(plot_format or "pdf").lower().strip().lstrip(".")
        if plot_format not in {"pdf", "png", "svg"}:
            plot_format = "pdf"

        belief = self.init_belief(
            N_p=N_p,
            sigma_init=sigma_init,
            init_bounds=init_bounds,
            init_x_range=init_x_range,
            init_y_range=init_y_range,
        )
        initial_summary = self.belief_support_summary(belief, tx_true_xy=tx_true_xy)
        if verbose and self.print_prior_debug:
            prior_bounds = self._tensor_to_bounds(belief.get("prior_bounds", None))
            print(
                "[prior] support="
                f"{self.format_bounds(prior_bounds)}, "
                f"effective x=[{initial_summary['x_min']:+.2f},{initial_summary['x_max']:+.2f}], "
                f"y=[{initial_summary['y_min']:+.2f},{initial_summary['y_max']:+.2f}], "
                f"mean=({initial_summary['x_mean']:+.2f},{initial_summary['y_mean']:+.2f}), "
                f"nearest_true={initial_summary.get('nearest_true_dist', float('nan')):.2f}m"
            )

        traj_rx = [rx_pose.detach().clone()]
        traj_est: List[torch.Tensor] = []
        traj_unc: List[torch.Tensor] = []
        traj_err: List[torch.Tensor] = []
        traj_ess: List[torch.Tensor] = []
        meas_history: List[torch.Tensor] = []
        pf_stats_history: List[Dict[str, float]] = []
        move_stats_history: List[Dict[str, float]] = []

        if plot_dir is not None and plot_every > 0:
            os.makedirs(plot_dir, exist_ok=True)
            self.plot_valid_mask(savepath=os.path.join(plot_dir, f"{self.rand_idx}_mask.{plot_format}"), show=False)
            est0 = self.particle_estimate(belief)
            self.plot_state(
                belief=belief,
                rx_traj=torch.stack(traj_rx, dim=0),
                estimate_xy=est0,
                tx_true_xy=tx_true_xy,
                title=f"ATDF-{fwd.upper()} step 0 (initial belief)",
                savepath=os.path.join(plot_dir, f"{self.rand_idx}_000.{plot_format}"),
                show=show_plots,
            )

        found = False
        success = False
        for t in range(1, T_max + 1):
            observed_rays = measurement_fn(rx_pose, tx_true_xy)
            meas_history.append(observed_rays.detach().clone())

            belief, pf_stats = self.particle_filter_update(
                belief=belief,
                observed_rays=observed_rays,
                rx_pose=rx_pose,
                likelihood_mode=likelihood_mode,
                forward_model=fwd,
                resample_threshold=resample_threshold,
                roughening_scale=roughening_scale,
                global_rejuvenation_frac=global_rejuvenation_frac,
            )
            if debug_truth_diagnostics:
                truth_diag = self.truth_likelihood_diagnostics(
                    rx_pose=rx_pose,
                    observed_rays=observed_rays,
                    belief=belief,
                    tx_true_xy=tx_true_xy,
                    likelihood_mode=likelihood_mode,
                    forward_model=fwd,
                )
                pf_stats.update(truth_diag)
            pf_stats_history.append(pf_stats)

            est_xy = self.particle_estimate(belief)
            unc = self.uncertainty_scalar(belief)
            err = float(torch.norm(est_xy - tx_true_xy).item())
            rx_to_est = float(torch.norm(rx_pose[:2] - est_xy).item())
            ess = self.effective_sample_size(belief)

            traj_est.append(est_xy.detach().clone())
            traj_unc.append(torch.tensor(unc, device=self.device))
            traj_err.append(torch.tensor(err, device=self.device))
            traj_ess.append(torch.tensor(ess, device=self.device))

            if verbose:
                msg = (
                    f"[t={t:03d}] est=({est_xy[0].item():+.3f},{est_xy[1].item():+.3f}) "
                    f"err={err:.3f} unc={unc:.3f} rx_to_est={rx_to_est:.3f} "
                    f"ESS={ess:.1f}/{N_p} rays={int(self.clean_rays(observed_rays).shape[0])} "
                    f"ll=[{pf_stats['ll_min']:.2f},{pf_stats['ll_max']:.2f}]"
                )
                if debug_truth_diagnostics and "true_ll_rank" in pf_stats:
                    msg += (
                        f" true_rank={int(pf_stats['true_ll_rank'])}/{N_p + 1} "
                        f"true_pct={pf_stats['true_ll_percentile']:.2f} "
                        f"mass1m={pf_stats['mass_within_1p0m']:.3f}"
                    )
                print(msg)

            if plot_dir is not None and plot_every > 0 and (t % plot_every == 0 or t == 1):
                self.plot_state(
                    belief=belief,
                    rx_traj=torch.stack(traj_rx, dim=0),
                    estimate_traj=torch.stack(traj_est, dim=0),
                    estimate_xy=est_xy,
                    tx_true_xy=tx_true_xy,
                    title=f"ATDF-{fwd.upper()} step {t}",
                    savepath=os.path.join(plot_dir, f"{self.rand_idx}_{t:03d}.{plot_format}"),
                    show=show_plots,
                )

            success = err <= success_error
            found_by_belief = (unc <= stop_uncertainty and rx_to_est <= stop_rx_to_est)
            found = found_by_belief or (bool(use_truth_success_stop) and success)
            if found:
                if verbose:
                    reason = "truth-success" if (success and bool(use_truth_success_stop) and not found_by_belief) else "uncertainty-and-range"
                    print(f"Stop at step {t}: {reason}; err={err:.3f}, unc={unc:.3f}, rx_to_est={rx_to_est:.3f}")
                break

            cand = self.generate_candidates(
                current_pose=rx_pose,
                estimate_xy=est_xy,
                num_pos=cand_num_pos,
                step=cand_step,
                num_headings=cand_num_headings,
            )
            if verbose:
                cdbg = getattr(self, "_last_candidate_debug", {})
                if cdbg:
                    print(
                        "  candidates: "
                        f"K={int(cdbg.get('num_candidates', cand.shape[0]))}, "
                        f"positions={int(cdbg.get('num_positions', 0))}, "
                        f"wallK={int(cdbg.get('num_wall_cross_candidates', 0))}, "
                        f"wall_step={float(cdbg.get('wall_cross_max_step', self.wall_cross_max_step)):.2f}m"
                    )

            if planning_mode.lower().strip() in {"mi", "hybrid"}:
                mi = self.mutual_information_candidates(belief, cand, mi_particle_limit=mi_particle_limit, forward_model=fwd)
            else:
                mi = torch.zeros(cand.shape[0], device=self.device)

            next_pose = self.select_next_pose(
                current_pose=rx_pose,
                cand=cand,
                mi=mi,
                estimate_xy=est_xy,
                uncertainty=unc,
                tx_true_xy=tx_true_xy,
            )
            move_stats = self.path_occupancy_stats(rx_pose[:2], next_pose[:2], role="rx")
            move_stats_history.append(move_stats)
            if verbose and move_stats["blocked_len"] > 1e-6:
                dbg = getattr(self, "_last_planning_debug", {})
                print(
                    "  selected wall-cross move: "
                    f"dist={move_stats['dist']:.2f}m, "
                    f"occ={move_stats['occupied_len']:.2f}m, "
                    f"unk={move_stats['unknown_len']:.2f}m, "
                    f"segments={int(round(move_stats['num_occupied_segments']))}, "
                    f"cost={move_stats['wall_cost']:.2f}, "
                    f"end_clear={move_stats.get('end_clearance', 0.0):.2f}m, "
                    f"score={float(dbg.get('score', 0.0)):.3f}, "
                    f"mi_norm={float(dbg.get('mi_norm', 0.0)):.3f}"
                )
            rx_pose = self.snap_rx_pose_to_valid(next_pose)
            traj_rx.append(rx_pose.detach().clone())

        est_final = self.particle_estimate(belief)
        _, cov_final = self.mixture_mean_and_cov(belief)

        return {
            "belief_w": belief["w"].detach(),
            "belief_mu": belief["mu"].detach(),
            "belief_Sigma": belief["Sigma"].detach(),
            "rx_traj": torch.stack(traj_rx, dim=0),
            "est_traj": torch.stack(traj_est, dim=0) if len(traj_est) > 0 else torch.empty(0, 2, device=self.device),
            "unc_traj": torch.stack(traj_unc, dim=0) if len(traj_unc) > 0 else torch.empty(0, device=self.device),
            "err_traj": torch.stack(traj_err, dim=0) if len(traj_err) > 0 else torch.empty(0, device=self.device),
            "ess_traj": torch.stack(traj_ess, dim=0) if len(traj_ess) > 0 else torch.empty(0, device=self.device),
            "measurement_rays": meas_history,
            "pf_stats": pf_stats_history,
            "move_stats": move_stats_history,
            "initial_belief_summary": initial_summary,
            "est_final": est_final.detach(),
            "cov_final": cov_final.detach(),
            "found": torch.tensor(bool(found), device=self.device),
            "success": torch.tensor(bool(success), device=self.device),
        }



def _ros_param(name: str, default):
    if ROS_AVAILABLE and rospy is not None:
        return rospy.get_param(name, default)
    return default


def _param_pose(prefix: str, default=None):
    """Read ROS params prefix/x, prefix/y, prefix/yaw_deg or prefix/yaw."""
    if not ROS_AVAILABLE or rospy is None:
        return default
    has_x = rospy.has_param(prefix + "/x")
    has_y = rospy.has_param(prefix + "/y")
    if not (has_x and has_y):
        return default
    x = float(rospy.get_param(prefix + "/x"))
    y = float(rospy.get_param(prefix + "/y"))
    if rospy.has_param(prefix + "/yaw_deg"):
        yaw = math.radians(float(rospy.get_param(prefix + "/yaw_deg")))
    else:
        yaw = float(rospy.get_param(prefix + "/yaw", 0.0))
    return torch.tensor([x, y, yaw], dtype=torch.float32)


def ros_main():
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS packages are unavailable. Source your ROS Noetic workspace and run with rosrun/roslaunch.")
    rospy.init_node("atdf_video", anonymous=False)

    config_path = _ros_param("~config_path", "config/config.yaml")
    map_yaml_path = _ros_param("~map_yaml_path", "maps/21202.yaml")
    checkpoint_path = _ros_param("~checkpoint_path", "checkpoints/epoch_0771.pth")
    rospy.loginfo(
        "[ATDF] Startup: script=%s; forward_model=%s; measurement_source=%s; checkpoint=%s",
        os.path.realpath(__file__), _ros_param("~forward_model", "nrp"),
        _ros_param("~measurement_source", "real_iq"), checkpoint_path,
    )

    atdf = ATDF(
        config_path=config_path,
        map_yaml_path=map_yaml_path,
        checkpoint_path=checkpoint_path,
        valid_pool_stride_px=int(_ros_param("~valid_pool_stride_px", 6)),
        rt_rx_chunk_size=int(_ros_param("~rt_rx_chunk_size", 8)),
        rt_tx_chunk_size=int(_ros_param("~rt_tx_chunk_size", 128)),
        nrp_batch_size=int(_ros_param("~nrp_batch_size", 512)),
        forward_model=str(_ros_param("~forward_model", "nrp")),
        measurement_source=str(_ros_param("~measurement_source", "real_iq")),
        restrict_to_nrp_domain=bool(_ros_param("~restrict_to_nrp_domain", True)),
        nrp_input_clamp=bool(_ros_param("~nrp_input_clamp", True)),
        antenna_offset_x_m=float(_ros_param("~antenna_offset_x_m", 0.0)),
        antenna_offset_y_m=float(_ros_param("~antenna_offset_y_m", -0.30)),
        antenna_yaw_offset_rad=float(_ros_param("~antenna_yaw_offset_rad", -0.5 * math.pi)),
        snap_forward_rx_to_valid=bool(_ros_param("~snap_forward_rx_to_valid", False)),
        real_iq_topic=str(_ros_param("~real_iq_topic", "isp_ble/data")),
        real_iq_num_measurements=int(_ros_param("~real_iq_num_measurements", 20)),
        real_iq_selected_indices=_ros_param("~real_iq_selected_indices", [7, 8, 9, 10]),
        real_iq_negative_frequency=bool(_ros_param("~real_iq_negative_frequency", True)),
        real_iq_reverse_array=bool(_ros_param("~real_iq_reverse_array", False)),
        real_iq_average_mode=str(_ros_param("~real_iq_average_mode", "mean_vector")),
        real_iq_normalize_snapshots=bool(_ros_param("~real_iq_normalize_snapshots", True)),
        real_iq_covariance_comparison=str(_ros_param("~real_iq_covariance_comparison", "correlation")),
        real_iq_cov_sigma=float(_ros_param("~real_iq_cov_sigma", 0.50)),
        real_iq_phase_offsets_rad=_ros_param("~real_iq_phase_offsets_rad", [0.0, 0.0, 0.0, 0.0]),
    )

    atdf.setup_ros_interfaces(
        robot_pose_source=str(_ros_param("~robot_pose_source", "auto")),
        robot_pose_topic=str(_ros_param("~robot_pose_topic", "/amcl_pose")),
        robot_pose_parent_frame=str(_ros_param("~robot_pose_parent_frame", _ros_param("~map_frame", "map"))),
        robot_pose_child_frame=str(_ros_param("~robot_pose_child_frame", _ros_param("~base_frame", "base_link"))),
        goal_topic=str(_ros_param("~goal_topic", "/move_base_simple/goal")),
        move_base_action_name=str(_ros_param("~move_base_action", "/move_base")),
        use_move_base_action=bool(_ros_param("~use_move_base_action", True)),
        goal_frame=str(_ros_param("~goal_frame", "map")),
        action_wait_timeout=float(_ros_param("~action_wait_timeout", 20.0)),
        robot_pose_max_age_s=float(_ros_param("~robot_pose_max_age_s", 2.0)),
        allow_goal_pose_fallback=bool(_ros_param("~allow_goal_pose_fallback", False)),
        # Optional legacy simulation topic; leave empty on the real robot.
        gt_pose_topic=str(_ros_param("~gt_pose_topic", "")),
        gt_tf_child_frame=str(_ros_param("~gt_tf_child_frame", _ros_param("~robot_pose_child_frame", _ros_param("~base_frame", "base_link")))),
        gt_tf_parent_frame=str(_ros_param("~gt_tf_parent_frame", _ros_param("~robot_pose_parent_frame", _ros_param("~map_frame", "map")))),
    )

    tx_true = None
    if rospy.has_param("~tx_true_x") and rospy.has_param("~tx_true_y"):
        tx_true = torch.tensor(
            [float(_ros_param("~tx_true_x", 0.0)), float(_ros_param("~tx_true_y", 0.0))],
            dtype=torch.float32,
        )
    rx_init = _param_pose("~rx_init", default=None)

    init_x_range = None
    init_y_range = None
    init_bounds = None
    if bool(_ros_param("~use_pf_bounds", False)):
        init_bounds = atdf._parse_xy_bounds([
            float(rospy.get_param("~pf_x_min")), float(rospy.get_param("~pf_x_max")),
            float(rospy.get_param("~pf_y_min")), float(rospy.get_param("~pf_y_max")),
        ])
        rospy.loginfo("[ATDF] Source PF search bounds: %s", atdf.format_bounds(init_bounds))
    else:
        # Preserve the previous optional prior parameters when the new limits
        # are disabled. Enabled PF bounds take precedence over these aliases.
        if rospy.has_param("~init_x_min") and rospy.has_param("~init_x_max"):
            init_x_range = (float(rospy.get_param("~init_x_min")), float(rospy.get_param("~init_x_max")))
        if rospy.has_param("~init_y_min") and rospy.has_param("~init_y_max"):
            init_y_range = (float(rospy.get_param("~init_y_min")), float(rospy.get_param("~init_y_max")))

    result = atdf.run_ros_active_localization(
        tx_true_xy=tx_true,
        rx_init_pose=rx_init,
        rx_init_reference=str(_ros_param("~rx_init_reference", "base")),
        N_p=int(_ros_param("~N_p", 1200)),
        sigma_init=float(_ros_param("~sigma_init", 2.5)),
        T_max=int(_ros_param("~T_max", 100)),
        stop_uncertainty=float(_ros_param("~stop_uncertainty", 0.45)),
        stop_rx_to_est=float(_ros_param("~stop_rx_to_est", 0.60)),
        success_error=float(_ros_param("~success_error", 0.75)),
        cand_num_pos=int(_ros_param("~cand_num_pos", 16)),
        cand_step=float(_ros_param("~cand_step", 0.75)),
        cand_num_headings=int(_ros_param("~cand_num_headings", 5)),
        mi_particle_limit=int(_ros_param("~mi_particle_limit", 96)),
        likelihood_mode=str(_ros_param("~likelihood_mode", "cov")),
        planning_mode=str(_ros_param("~planning_mode", "hybrid")),
        forward_model=str(_ros_param("~forward_model", "nrp")),
        measurement_source=str(_ros_param("~measurement_source", "real_iq")),
        resample_threshold=float(_ros_param("~resample_threshold", 0.50)),
        roughening_scale=float(_ros_param("~roughening_scale", 0.10)),
        global_rejuvenation_frac=float(_ros_param("~global_rejuvenation_frac", 0.02)),
        init_bounds=init_bounds,
        init_x_range=init_x_range,
        init_y_range=init_y_range,
        debug_truth_diagnostics=bool(_ros_param("~debug_truth_diagnostics", False)),
        use_truth_success_stop=bool(_ros_param("~use_truth_success_stop", False)),
        truth_indicator_only=bool(_ros_param("~truth_indicator_only", True)),
        goal_timeout=float(_ros_param("~goal_timeout", 300.0)),
        goal_xy_tolerance=float(_ros_param("~goal_xy_tolerance", 0.25)),
        goal_yaw_tolerance=float(_ros_param("~goal_yaw_tolerance", 0.30)),
        plot_every=int(_ros_param("~plot_every", 1)),
        plot_dir=_ros_param("~plot_dir", None),
        plot_format=str(_ros_param("~plot_format", "pdf")),
        show_plots=bool(_ros_param("~show_plots", False)),
        verbose=bool(_ros_param("~verbose", True)),
    )
    rospy.loginfo(f"[ATDF] Final estimate: {result['est_final'].detach().cpu().numpy()}")
    rospy.loginfo(f"[ATDF] Found={bool(result['found'].item())}, Success={bool(result['success'].item())}")
    if bool(_ros_param("~keep_alive_after_finish", True)):
        rospy.loginfo("[ATDF] Run complete; retaining final RViz markers until shutdown.")
        rospy.spin()


if __name__ == "__main__":
    ros_main()
