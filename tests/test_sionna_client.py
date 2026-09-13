"""Exercise the legacy file exchange without ROS, Sionna, or a GPU."""

import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/atdf_video/src/atdf_sionna.py"
SPEC = importlib.util.spec_from_file_location("atdf_sionna", str(MODULE_PATH))
SIONNA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIONNA)


class SionnaClientTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.directory = Path(self.tempdir.name)

    def worker(self, ignored_ids=(), malformed=False):
        """Mirror worker ordering: read ID, poses, write response, commit ack."""
        errors = []

        def work():
            try:
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        run_id = int(np.load(str(self.directory / "test_run_id.npy")))
                    except (OSError, ValueError, EOFError):
                        time.sleep(0.001)
                        continue
                    if run_id in ignored_ids:
                        time.sleep(0.001)
                        continue
                    with np.load(str(self.directory / "test_pose_data.npz")) as data:
                        rx = data["rx_pose"]
                        tx = data["tx_pose"]
                    pairs = [(r, t) for r in rx for t in tx]
                    response = {"batch_u": np.array([np.concatenate([t, r]) for r, t in pairs])}
                    for index in range(len(pairs)):
                        response["pair_{}".format(index)] = np.full((index + 1, 6), index, dtype=np.float32)
                    if malformed:
                        response["pair_0"] = np.zeros((1, 5), dtype=np.float32)
                    np.savez(str(self.directory / "test_data.npz"), **response)
                    np.save(str(self.directory / "test_run_id_update_temp.npy"), run_id)
                    os.replace(str(self.directory / "test_run_id_update_temp.npy"),
                               str(self.directory / "test_run_id_update.npy"))
                    return
                raise RuntimeError("Fake worker received no request")
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread, errors

    def assert_worker_finished(self, thread, errors):
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_missing_ack_startup_and_ragged_cross_product_order(self):
        rx = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tx = np.array([[5, 6], [7, 8]], dtype=np.float32)
        thread, errors = self.worker()
        client = SIONNA.SionnaClient(self.directory, timeout_s=2, poll_interval_s=0.001)
        batch_u, rays = client.inquire_ray(rx, tx)
        self.assert_worker_finished(thread, errors)
        self.assertEqual(batch_u.dtype, torch.float32)
        np.testing.assert_array_equal(batch_u.numpy(), [[5, 6, 1, 2], [7, 8, 1, 2],
                                                     [5, 6, 3, 4], [7, 8, 3, 4]])
        self.assertEqual([tuple(ray.shape) for ray in rays], [(1, 6), (2, 6), (3, 6), (4, 6)])
        for index, ray in enumerate(rays):
            self.assertTrue(torch.all(ray == index).item())

    def test_existing_request_and_stale_ack_ids_are_not_reused(self):
        np.save(str(self.directory / "test_run_id.npy"), 41)
        np.save(str(self.directory / "test_run_id_update.npy"), 42)
        # A stale response must not be consumed merely because it exists.
        np.savez(str(self.directory / "test_data.npz"), batch_u=np.array([[99] * 4]))
        thread, errors = self.worker(ignored_ids=(41,))
        client = SIONNA.SionnaClient(self.directory, timeout_s=2, poll_interval_s=0.001)
        with mock.patch.object(SIONNA.secrets, "randbits", side_effect=[41, 42, 43]):
            batch_u, rays = client.inquire_ray([[1, 2]], [[3, 4]])
        self.assert_worker_finished(thread, errors)
        self.assertEqual(int(np.load(str(self.directory / "test_run_id.npy"))), 43)
        np.testing.assert_array_equal(batch_u.numpy(), [[3, 4, 1, 2]])
        self.assertEqual(len(rays), 1)

    def test_missing_worker_times_out_without_accepting_stale_ack(self):
        np.save(str(self.directory / "test_run_id_update.npy"), 42)
        client = SIONNA.SionnaClient(self.directory, timeout_s=0.03, poll_interval_s=0.001)
        with self.assertRaisesRegex(TimeoutError, "separate Python environment"):
            client.inquire_ray([[1, 2]], [[3, 4]])

    def test_shutdown_during_wait_cancels_query(self):
        calls = []

        def shutting_down():
            calls.append(True)
            return len(calls) > 3

        client = SIONNA.SionnaClient(self.directory, timeout_s=2, poll_interval_s=0.001,
                                    is_shutdown=shutting_down)
        with self.assertRaisesRegex(RuntimeError, "cancelled because shutdown"):
            client.inquire_ray([[1, 2]], [[3, 4]])

    def test_acknowledged_malformed_response_is_rejected(self):
        thread, errors = self.worker(malformed=True)
        client = SIONNA.SionnaClient(self.directory, timeout_s=2, poll_interval_s=0.001)
        with self.assertRaisesRegex(RuntimeError, "pair_0 must have finite shape"):
            client.inquire_ray([[1, 2]], [[3, 4]])
        self.assert_worker_finished(thread, errors)


if __name__ == "__main__":
    unittest.main()
