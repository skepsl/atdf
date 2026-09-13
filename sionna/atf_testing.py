#!/usr/bin/env python3
"""Independent Sionna RT worker for the ROS file-exchange protocol.

Run this script in the separate Sionna environment. The measurement and path
normalization settings match ``additional/atf_testing.py`` from the experiment
workspace; only startup and the file-exchange handling are made portable.
"""

import argparse
import math
import os
from pathlib import Path
import tempfile
import time

P_MAX = 20
D_MAX = 15
MAX_DB = -40
MIN_DB = -120
TAU_MIN = 0.0
TAU_MAX = 0.000001


class FileExchange:
    """Single-client exchange: poses/request in, atomic data/ack out."""

    def __init__(self, database_dir):
        import numpy as np
        self.np = np
        self.directory = Path(database_dir).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def read_id(self, filename):
        try:
            value = self.np.load(self.directory / filename, allow_pickle=False)
            if value.size != 1 or value.dtype.kind not in "iu":
                return None
            return int(value.reshape(-1)[0])
        except (OSError, ValueError, EOFError):
            return None

    def read_request(self, last_completed=None):
        run_id = self.read_id("test_run_id.npy")
        if run_id is None or run_id < 0 or run_id == last_completed:
            return None
        try:
            with self.np.load(self.directory / "test_pose_data.npz", allow_pickle=False) as data:
                rx_pose = data["rx_pose"].copy()
                tx_pose = data["tx_pose"].copy()
        except FileNotFoundError:
            # A completely new exchange directory needs no seed files.
            return None
        for label, poses in (("rx_pose", rx_pose), ("tx_pose", tx_pose)):
            if (poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 2
                    or poses.dtype.kind not in "iuf" or not self.np.isfinite(poses).all()):
                raise ValueError("{} must be a finite, nonempty [N, 2] array".format(label))
        if self.read_id("test_run_id.npy") != run_id:
            return None
        return run_id, rx_pose, tx_pose

    def _atomic_save(self, filename, value, archive=False):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=str(self.directory),
                                             prefix="." + filename + ".", delete=False) as handle:
                temporary = handle.name
                if archive:
                    self.np.savez(handle, **value)
                else:
                    self.np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, str(self.directory / filename))
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def write_response(self, run_id, batch_data):
        # An acknowledgement becomes visible only after the full response.
        self._atomic_save("test_data.npz", batch_data, archive=True)
        self._atomic_save("test_run_id_update.npy", self.np.asarray(run_id, dtype=self.np.int64))


class Raygen:
    def __init__(self, scene_path, database_dir="~/atdf_database", gpu="0", poll_interval=0.1):
        self.database_dir = str(Path(database_dir).expanduser().resolve())
        self.poll_interval = float(poll_interval)
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive and finite")
        scene_path = Path(scene_path).expanduser().resolve()
        if not scene_path.is_file():
            raise FileNotFoundError("Sionna scene XML does not exist: {}".format(scene_path))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

        # Import only after command-line parsing and GPU selection. --help
        # works without installing Sionna, NumPy, or einops.
        from sionna.rt import LambertianPattern, load_scene, Transmitter, Receiver, PlanarArray, PathSolver
        self._Receiver = Receiver
        self._Transmitter = Transmitter
        self.exchange = FileExchange(self.database_dir)
        self.max_depth = 10
        self.num_samples = 600000
        self.scene = load_scene(str(scene_path))
        self.scene.frequency = 24e8
        scattering_pattern = LambertianPattern()
        itu_concrete_mat = self.scene.get("itu_concrete")
        if itu_concrete_mat is None:
            raise ValueError("Scene must provide the reference radio material 'itu_concrete'")
        itu_concrete_mat.scattering_pattern = scattering_pattern
        itu_concrete_mat.alpha_r = 1
        itu_concrete_mat.scattering_coefficient = 0.6
        self.scene.synthetic_array = True
        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                         vertical_spacing=0.5, horizontal_spacing=0.5,
                                         pattern="iso", polarization="V")
        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                         vertical_spacing=0.5, horizontal_spacing=0.5,
                                         pattern="iso", polarization="V")
        self.p_solver = PathSolver()

    def measure(self, rx_pose, tx_pose):
        import numpy as np
        added = []
        try:
            for r in range(rx_pose.shape[0]):
                rx_loc = np.asarray([rx_pose[r][0], rx_pose[r][1], 0.5])
                rx = self._Receiver(name=f"rx_{r}", position=rx_loc, orientation=[.0, .0, .0])
                self.scene.add(rx)
                added.append(f"rx_{r}")
            for t in range(tx_pose.shape[0]):
                tx_loc = np.asarray([tx_pose[t][0], tx_pose[t][1], 0.5])
                tx = self._Transmitter(name=f"tx_{t}", position=tx_loc, orientation=[.0, .0, .0])
                self.scene.add(tx)
                added.append(f"tx_{t}")
            return self._evaluate_paths(rx_pose, tx_pose)
        finally:
            for name in added:
                self.scene.remove(name)
            self.scene.remove("paths")

    def _evaluate_paths(self, rx_pose, tx_pose):
        import numpy as np
        from einops import rearrange
        paths = self.p_solver(scene=self.scene,
                         max_depth=8,
                         max_num_paths_per_src=2500,
                         los=True,
                         specular_reflection=True,
                         diffuse_reflection=True,
                         refraction=True,
                         diffraction=True,
                         edge_diffraction=True,
                         diffraction_lit_region=True,
                         synthetic_array=False,
                         seed=41)
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        a = a[:, :, :, :, :, 0]

        # Amplitude
        a_abs_dB = 20 * np.log10(np.abs(a) + 1e-12)
        a_clipped = np.clip(a_abs_dB, a_min=MIN_DB, a_max=MAX_DB)
        a_norm = (a_clipped - MIN_DB) / (MAX_DB - MIN_DB)

        # Tau
        tau_clipped = np.clip(tau, a_min=0.0, a_max=TAU_MAX)
        tau_norm = tau_clipped / TAU_MAX

        # Angles
        theta_t_norm = paths.theta_t.numpy()
        theta_r_norm = paths.theta_r.numpy()
        phi_t_norm = paths.phi_t.numpy()
        phi_r_norm = paths.phi_r.numpy()

        valid = paths.valid.numpy()
        mask = valid.astype(np.uint8)

        # ==========================================
        # 2. STACK AND SORT
        # ==========================================
        features_raw = np.stack([
            a_norm, tau_norm, theta_t_norm, phi_t_norm, theta_r_norm, phi_r_norm
        ], axis=-1)

        # Mask using the normalized amplitude!
        a_abs_masked = np.where(mask == 1, a_norm, -1.0)
        R, Nr, T, Nt, P_current = a_norm.shape[:5]
        P_keep = min(P_MAX, P_current)

        top_idx = np.argsort(a_abs_masked, axis=4)[..., -P_keep:][..., ::-1]

        idx_expanded = np.repeat(np.expand_dims(top_idx, axis=-1), 6, axis=-1)
        features_top = np.take_along_axis(features_raw, idx_expanded, axis=4)
        mask_top = np.take_along_axis(mask, top_idx, axis=4)

        # ==========================================
        # 3. HANDLE PADDING
        # ==========================================
        if P_current < P_MAX:
            pad_size = P_MAX - P_current

            pad_feat = ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_size), (0, 0))
            features_top = np.pad(features_top, pad_feat, constant_values=0.0)

            pad_mask = ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_size))
            mask_top = np.pad(mask_top, pad_mask, constant_values=0)

        # ==========================================
        # 4. SQUASH TO BATCH
        # ==========================================
        features_clean = rearrange(features_top, 'r 1 t 1 p c -> (r t) p c')
        mask_clean = rearrange(mask_top, 'r 1 t 1 p -> (r t) p')

        # ==========================================
        # 5. RAGGED LIST & BATCH_U CONSTRUCTION
        # ==========================================
        valid_mask = (mask_clean == 1)
        b_size = features_clean.shape[0]
        T_len = len(tx_pose)

        # Map boundaries
        MAP_X_MIN, MAP_X_MAX = -13.0, 6.0
        MAP_Y_MIN, MAP_Y_MAX = -2.5, 10.0

        # Initialize your lists and arrays
        features_list = []
        batch_u = np.zeros((b_size, 4), dtype=np.float32)

        for i in range(b_size):
            # Recover the specific Tx and Rx index for this batch item
            r_idx = i // T_len
            t_idx = i % T_len

            # Build the raw [tx_x, tx_y, rx_x, rx_y] vector
            tx_xy = tx_pose[t_idx][:2]
            rx_xy = rx_pose[r_idx][:2]
            batch_u[i] = np.concatenate([tx_xy, rx_xy])

            # Sift out the padded/invalid paths
            valid_paths = features_clean[i][valid_mask[i]]

            # If 0 paths, insert dummy zero-path
            if len(valid_paths) == 0:
                valid_paths = np.zeros((1, 6), dtype=features_clean.dtype)

            features_list.append(valid_paths)

        # ==========================================
        # 5.5 NORMALIZE BATCH_U TO [0.0, 1.0]
        # ==========================================
        # Normalize all X coordinates (Indices 0 and 2: tx_x, rx_x)
        batch_u[:, [0, 2]] = (batch_u[:, [0, 2]] - MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN)

        # Normalize all Y coordinates (Indices 1 and 3: tx_y, rx_y)
        batch_u[:, [1, 3]] = (batch_u[:, [1, 3]] - MAP_Y_MIN) / (MAP_Y_MAX - MAP_Y_MIN)

        # ==========================================
        # 6. ATOMIC SAVE & RETURN
        # ==========================================
        # 1. Pack the jagged list of paths into the dictionary
        batch_data = {f'pair_{i}': features_list[i] for i in range(b_size)}

        # 2. Add your [b, 4] input tensor to the same dictionary!
        batch_data['batch_u'] = batch_u

        return batch_data

    def synthesize(self, once=False):
        old_run_id = self.exchange.read_id("test_run_id_update.npy")
        print("Waiting for requests in {}".format(self.database_dir), flush=True)
        while True:
            request = self.exchange.read_request(last_completed=old_run_id)
            if request is None:
                time.sleep(self.poll_interval)
                continue
            run_id, rx_pose, tx_pose = request
            print("execute {}".format(run_id), flush=True)
            batch_data = self.measure(rx_pose, tx_pose)
            self.exchange.write_response(run_id, batch_data)
            old_run_id = run_id
            print("Done {}".format(run_id), flush=True)
            if once:
                return


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path,
                        help="Scene XML exported with its referenced meshes and the itu_concrete material")
    parser.add_argument("--database-dir", default="~/atdf_database",
                        help="Exchange directory shared with the ROS client (default: %(default)s)")
    parser.add_argument("--gpu", default="0",
                        help="CUDA_VISIBLE_DEVICES value, set before Sionna imports (default: %(default)s)")
    parser.add_argument("--poll-interval", type=positive_seconds, default=0.1,
                        help="Seconds between idle request checks (default: %(default)s)")
    parser.add_argument("--once", action="store_true",
                        help="Wait for and answer one unacknowledged request, then exit")
    args = parser.parse_args(argv)
    worker = Raygen(args.scene, args.database_dir, args.gpu, args.poll_interval)
    try:
        worker.synthesize(once=args.once)
    except KeyboardInterrupt:
        print("Sionna worker stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
