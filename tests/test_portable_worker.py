"""Portable worker startup and file protocol, without Sionna or a simulator."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "sionna/atf_testing.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = load_module("portable_sionna_worker", WORKER_PATH)


class PortableWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "exchange"
        self.exchange = WORKER.FileExchange(self.directory)
        self.rx = np.array([[1.0, 2.0]], dtype=np.float32)
        self.tx = np.array([[3.0, 4.0]], dtype=np.float32)
        self.response = {
            "pair_0": np.arange(18, dtype=np.float32).reshape(3, 6),
            "batch_u": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        }

    def write_request(self, run_id, rx=None, tx=None):
        self.exchange._atomic_save("test_pose_data.npz", {
            "rx_pose": self.rx if rx is None else rx,
            "tx_pose": self.tx if tx is None else tx,
        }, archive=True)
        self.exchange._atomic_save("test_run_id.npy", np.int64(run_id))

    def fake_worker(self):
        # Bypass scene construction only; execute the real file protocol loop.
        worker = WORKER.Raygen.__new__(WORKER.Raygen)
        worker.database_dir = str(self.directory)
        worker.exchange = self.exchange
        worker.poll_interval = 0.001
        worker.measure = mock.Mock(return_value=self.response)
        return worker

    def test_fresh_directory_needs_no_seed_files(self):
        self.assertTrue(self.directory.is_dir())
        self.assertIsNone(self.exchange.read_request())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_idle_request_is_ignored_even_with_pose_data(self):
        for run_id in (-1, -9):
            with self.subTest(run_id=run_id):
                self.write_request(run_id)
                self.assertIsNone(self.exchange.read_request())

    def test_request_without_pose_file_waits(self):
        self.exchange._atomic_save("test_run_id.npy", np.int64(123))
        self.assertIsNone(self.exchange.read_request())

    def test_large_request_id_and_coordinates_survive_loading(self):
        run_id = (1 << 62) + 12345
        self.write_request(run_id)
        loaded_id, rx, tx = self.exchange.read_request()
        self.assertEqual(loaded_id, run_id)
        np.testing.assert_array_equal(rx, self.rx)
        np.testing.assert_array_equal(tx, self.tx)
        self.assertIsNone(self.exchange.read_request(last_completed=run_id))

    def test_changed_request_id_during_load_is_not_processed(self):
        self.write_request(123)
        with mock.patch.object(self.exchange, "read_id", side_effect=[123, 124]):
            self.assertIsNone(self.exchange.read_request())

    def test_invalid_ids_are_ignored_without_loading_pickle(self):
        for value in (np.float64(12), np.array([12, 13]),
                      np.array({"run_id": 12}, dtype=object)):
            with self.subTest(value=value):
                np.save(str(self.directory / "test_run_id.npy"), value)
                self.assertIsNone(self.exchange.read_request())

    def test_invalid_pose_arrays_are_rejected(self):
        for poses in (np.zeros((1, 3)), np.zeros((0, 2)),
                      np.array([[1.0, np.nan]]), np.ones((1, 2), dtype=complex)):
            with self.subTest(shape=poses.shape, dtype=poses.dtype):
                self.write_request(123, rx=poses)
                with self.assertRaisesRegex(ValueError, "rx_pose"):
                    self.exchange.read_request()

    def test_complete_response_is_visible_before_acknowledgement(self):
        replace = WORKER.os.replace
        committed = []

        def observe_replace(source, destination):
            destination = Path(destination)
            self.assertEqual(destination.parent, self.directory)
            if destination.name == "test_run_id_update.npy":
                with np.load(self.directory / "test_data.npz", allow_pickle=False) as data:
                    for name, expected in self.response.items():
                        np.testing.assert_array_equal(data[name], expected)
                self.assertIsNone(self.exchange.read_id("test_run_id_update.npy"))
            committed.append(destination.name)
            replace(source, destination)

        with mock.patch.object(WORKER.os, "replace", side_effect=observe_replace):
            self.exchange.write_response(123, self.response)
        self.assertEqual(committed, ["test_data.npz", "test_run_id_update.npy"])
        self.assertEqual(self.exchange.read_id("test_run_id_update.npy"), 123)
        self.assertFalse(any(path.name.startswith(".") for path in self.directory.iterdir()))

    def test_failed_response_write_preserves_previous_data_and_ack(self):
        self.exchange.write_response(122, self.response)
        original = (self.directory / "test_data.npz").read_bytes()

        def interrupted_save(handle, **arrays):
            handle.write(b"incomplete archive")
            raise OSError("simulated write failure")

        with mock.patch.object(self.exchange.np, "savez", side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                self.exchange.write_response(123, self.response)
        self.assertEqual((self.directory / "test_data.npz").read_bytes(), original)
        self.assertEqual(self.exchange.read_id("test_run_id_update.npy"), 122)
        self.assertFalse(any(path.name.startswith(".") for path in self.directory.iterdir()))

    def test_restart_processes_new_request_despite_old_ack(self):
        self.exchange.write_response(122, self.response)
        self.write_request(123)
        worker = self.fake_worker()
        worker.synthesize(once=True)
        worker.measure.assert_called_once()
        np.testing.assert_array_equal(worker.measure.call_args[0][0], self.rx)
        np.testing.assert_array_equal(worker.measure.call_args[0][1], self.tx)
        self.assertEqual(self.exchange.read_id("test_run_id_update.npy"), 123)

    def test_restart_waits_when_existing_request_is_already_acknowledged(self):
        self.write_request(123)
        self.exchange.write_response(123, self.response)
        worker = self.fake_worker()
        with mock.patch.object(WORKER.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.synthesize(once=True)
        worker.measure.assert_not_called()
        self.assertEqual(self.exchange.read_id("test_run_id_update.npy"), 123)

    def test_real_client_completes_request_with_portable_worker(self):
        client_module = load_module("portable_worker_client",
                                    ROOT / "src/atdf_video/src/atdf_sionna.py")
        worker = self.fake_worker()
        stopped = threading.Event()
        read_request = worker.exchange.read_request
        errors = []

        class StopWorker(Exception):
            pass

        def cancellable_read(*args, **kwargs):
            if stopped.is_set():
                raise StopWorker()
            return read_request(*args, **kwargs)

        worker.exchange.read_request = cancellable_read

        def run_worker():
            try:
                worker.synthesize(once=True)
            except StopWorker:
                pass
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run_worker, daemon=True)
        thread.start()
        try:
            client = client_module.SionnaClient(self.directory, timeout_s=2.0,
                                                poll_interval_s=0.001)
            batch, rays = client.inquire_ray(self.rx, self.tx)
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            np.testing.assert_array_equal(batch.numpy(), self.response["batch_u"])
            np.testing.assert_array_equal(rays[0].numpy(), self.response["pair_0"])
            self.assertEqual(self.exchange.read_id("test_run_id.npy"),
                             self.exchange.read_id("test_run_id_update.npy"))
        finally:
            stopped.set()
            thread.join(timeout=2.0)

    def test_help_runs_with_only_the_standard_library(self):
        # -S disables site packages, including NumPy and Sionna.
        result = subprocess.run([sys.executable, "-S", str(WORKER_PATH), "--help"],
                                capture_output=True, text=True, timeout=5.0)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--scene", "--database-dir", "--gpu", "--once"):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
