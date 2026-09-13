"""File client for the separate Sionna environment in sionna/atf_testing.py.

The existing worker owns ``test_data.npz`` and its acknowledgement ID. This
client commits the pose data before atomically replacing the request ID, then
waits for that exact acknowledgement. Run only one ATDF client per exchange
directory; the legacy worker protocol has a single request/response slot.
"""

import os
import secrets
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


class SionnaClient:
    def __init__(self, database_dir="~/atdf_database",
                 timeout_s=120.0, poll_interval_s=0.1, is_shutdown=None):
        self.database_dir = Path(database_dir).expanduser()
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Sionna timeout_s must be positive and finite")
        if not np.isfinite(self.poll_interval_s) or self.poll_interval_s <= 0:
            raise ValueError("Sionna poll_interval_s must be positive and finite")
        self.is_shutdown = is_shutdown or (lambda: False)

    def _read_id(self, filename):
        try:
            value = np.load(str(self.database_dir / filename), allow_pickle=False)
            if value.size != 1 or value.dtype.kind not in "iu":
                return None
            return int(value.reshape(-1)[0])
        except (OSError, ValueError, EOFError):
            # A worker may not have produced any acknowledgement yet. Legacy
            # startup scripts can also leave an empty file until their first run.
            return None

    def _atomic_save(self, filename, value, archive=False):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", dir=str(self.database_dir),
                    prefix="." + filename + ".", delete=False) as handle:
                temporary = handle.name
                if archive:
                    np.savez(handle, **value)
                else:
                    np.save(handle, value, allow_pickle=False)
            os.replace(temporary, str(self.database_dir / filename))
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    @staticmethod
    def _validate_poses(poses, label):
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 2 or poses.shape[0] == 0:
            raise ValueError("{} must have nonempty shape [N, 2]".format(label))
        if not np.all(np.isfinite(poses)):
            raise ValueError("{} must contain finite coordinates".format(label))
        return poses

    def inquire_ray(self, rx_np, tx_np):
        """Return normalized [R*T,4] inputs and r-major [P,6] ray tensors."""
        rx_np = self._validate_poses(rx_np, "rx_np")
        tx_np = self._validate_poses(tx_np, "tx_np")
        if self.is_shutdown():
            raise RuntimeError("Sionna query cancelled because shutdown was requested")
        self.database_dir.mkdir(parents=True, exist_ok=True)
        excluded_ids = {
            self._read_id("test_run_id.npy"),
            self._read_id("test_run_id_update.npy"),
        }
        run_id = secrets.randbits(63)
        while run_id in excluded_ids:
            run_id = secrets.randbits(63)

        # The unchanged worker first reads the request ID, then the pose data.
        # Commit the complete archive before advertising this new request ID.
        self._atomic_save("test_pose_data.npz", {"rx_pose": rx_np, "tx_pose": tx_np}, archive=True)
        self._atomic_save("test_run_id.npy", np.asarray(run_id, dtype=np.int64))

        deadline = time.monotonic() + self.timeout_s
        while True:
            if self.is_shutdown():
                raise RuntimeError("Sionna query {} cancelled because shutdown was requested".format(run_id))
            if self._read_id("test_run_id_update.npy") == run_id:
                return self._load_response(rx_np.shape[0] * tx_np.shape[0], run_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Sionna worker did not acknowledge query {} within {:.1f}s in {}. "
                    "Start sionna/atf_testing.py in its separate Python environment "
                    "and use the same database directory.".format(
                        run_id, self.timeout_s, self.database_dir))
            time.sleep(min(self.poll_interval_s, remaining))

    def _load_response(self, pair_count, run_id):
        try:
            with np.load(str(self.database_dir / "test_data.npz"), allow_pickle=False) as data:
                batch_u = np.asarray(data["batch_u"], dtype=np.float32)
                if batch_u.shape != (pair_count, 4) or not np.all(np.isfinite(batch_u)):
                    raise ValueError("batch_u must have finite shape [{}, 4]".format(pair_count))
                batch_targets = []
                for index in range(pair_count):
                    paths = np.asarray(data["pair_{}".format(index)], dtype=np.float32)
                    if paths.ndim != 2 or paths.shape[1] != 6 or not np.all(np.isfinite(paths)):
                        raise ValueError("pair_{} must have finite shape [P, 6]".format(index))
                    batch_targets.append(torch.from_numpy(paths.copy()))
                return torch.from_numpy(batch_u.copy()), batch_targets
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise RuntimeError("Invalid Sionna response for query {}: {}".format(run_id, exc)) from exc
