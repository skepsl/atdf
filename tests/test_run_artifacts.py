"""Verify experiment snapshots survive independently of rendering."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/atdf_video/src"
sys.path.insert(0, str(SOURCE_DIR))
from atdf_artifacts import RunArtifacts


class RunArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_run_directories_are_unique_even_with_same_id_and_time(self):
        with mock.patch("atdf_artifacts._utc_now", return_value="2026-09-11T00:00:00+00:00"):
            first = RunArtifacts(self.root, "experiment/../../42")
            second = RunArtifacts(self.root, "experiment/../../42")
        self.assertNotEqual(first.directory, second.directory)
        self.assertEqual(first.directory.parent, self.root)
        for run in (first, second):
            self.assertTrue((run.directory / "metadata.json").is_file())

    def test_full_array_roundtrip_preserves_dtype_shape_and_values(self):
        run = RunArtifacts(self.root, 42, {"source": "sionna"})
        arrays = {
            "mu": np.arange(10, dtype=np.float32).reshape(5, 2),
            "Sigma": np.eye(2, dtype=np.float64),
            "measurement": np.empty((0, 6), dtype=np.float32),
            "measurement_stamps": np.array([123], dtype=np.int64),
            "error": np.array(np.nan),
        }
        path = run.save_step(np.int64(0), arrays, {"actual_pose": [1, 2, 3]})
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(arrays))
            for key, expected in arrays.items():
                np.testing.assert_array_equal(saved[key], expected)
                self.assertEqual(saved[key].dtype, expected.dtype)
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["step"], 0)
        self.assertEqual(metadata["run_id"], "42")
        self.assertEqual(metadata["metadata"]["actual_pose"], [1, 2, 3])

    def test_metadata_contains_schema_utc_and_json_safe_values(self):
        run = RunArtifacts(self.root, "run1", {
            "nested": [np.float32("nan"), float("inf"), {"n": np.int64(2)}],
            "limits": np.array([1., 2.]), "path": self.root,
        })
        content = (run.directory / "metadata.json").read_text()
        self.assertNotIn("NaN", content)
        metadata = json.loads(content)
        self.assertEqual(metadata["schema_version"], 1)
        self.assertTrue(metadata["created_utc"].endswith("+00:00"))
        self.assertEqual(metadata["metadata"]["nested"], [None, None, {"n": 2}])
        self.assertEqual(metadata["metadata"]["limits"], [1., 2.])

    def test_duplicate_step_never_overwrites_first_result(self):
        run = RunArtifacts(self.root, "run1")
        path = run.save_step(1, {"mu": np.array([1.])})
        original = path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            run.save_step(1, {"mu": np.array([2.])})
        self.assertEqual(path.read_bytes(), original)

    def test_failed_array_write_removes_temporary_file_and_reports_path(self):
        run = RunArtifacts(self.root, "run1")
        def fail_write(handle, **arrays):
            handle.write(b"partial")
            raise OSError("disk full")
        with mock.patch("atdf_artifacts.np.savez", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "Cannot save artifact step 1.*disk full"):
                run.save_step(1, {"mu": np.array([1.])})
        self.assertEqual(sorted(p.name for p in run.directory.iterdir()), ["metadata.json"])

    def test_failed_metadata_commit_keeps_raw_arrays_without_completion_marker(self):
        run = RunArtifacts(self.root, "run1")
        real_link = __import__("os").link
        def fail_metadata_link(source, destination):
            if destination.endswith("step_000001.json"):
                raise OSError("disk full")
            return real_link(source, destination)
        with mock.patch("atdf_artifacts.os.link", side_effect=fail_metadata_link):
            with self.assertRaisesRegex(OSError, "disk full"):
                run.save_step(1, {"mu": np.array([1.])})
        self.assertEqual(sorted(p.name for p in run.directory.iterdir()),
                         ["metadata.json", "step_000001.npz"])
        with self.assertRaises(FileExistsError):
            run.save_step(1, {"mu": np.array([2.])})

    def test_invalid_steps_and_pickle_arrays_are_rejected(self):
        run = RunArtifacts(self.root, "run1")
        for step in (-1, 1.0, True, "1"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                run.save_step(step, {"mu": np.array([1.])})
        for value in (np.array([{}], dtype=object), [1, 2]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "without object dtype"):
                run.save_step(1, {"mu": value})


if __name__ == "__main__":
    unittest.main()
