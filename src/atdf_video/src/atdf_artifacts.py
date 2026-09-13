"""Durable, pickle-free experiment snapshots, independent of plot settings."""

from datetime import datetime, timezone
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    """Keep metadata valid JSON, representing unavailable numbers as null."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Unsupported run metadata type: {}".format(type(value).__name__))


def _json_bytes(value):
    return (json.dumps(_json_safe(value), indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


class RunArtifacts:
    """One unique run directory containing metadata and complete step arrays.

    ``directory`` is a :class:`pathlib.Path`. A step's JSON file is committed
    after its NPZ file, so readers can use the JSON as its completion marker.
    Existing files are never replaced, including after an interrupted write.
    """

    SCHEMA_VERSION = 1

    def __init__(self, output_dir, run_id, metadata=None):
        self.run_id = str(run_id)
        created = _utc_now()
        document = _json_bytes({
            "schema_version": self.SCHEMA_VERSION,
            "created_utc": created,
            "run_id": self.run_id,
            "metadata": metadata or {},
        })
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", self.run_id)[:100] or "unnamed"
        timestamp = re.sub(r"[^0-9]", "", created.split("+")[0]) + "Z"
        root = Path(output_dir).expanduser()
        base_name = "run_{}_{}".format(timestamp, safe_id)
        try:
            root.mkdir(parents=True, exist_ok=True)
            suffix = 0
            while True:
                self.directory = root / (base_name + ("_{}".format(suffix) if suffix else ""))
                try:
                    self.directory.mkdir()
                    break
                except FileExistsError:
                    suffix += 1
            self._write_atomic(self.directory / "metadata.json",
                               lambda handle: handle.write(document))
        except OSError as exc:
            raise OSError("Cannot initialize run artifacts in {}: {}".format(root, exc)) from exc

    def _write_atomic(self, destination, writer):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=str(self.directory),
                                             prefix="." + destination.name + ".",
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard link publishes the fully written file atomically while
            # refusing to replace an existing result (unlike os.replace).
            os.link(str(temporary), str(destination))
            temporary.unlink()
            temporary = None
            directory_fd = os.open(str(self.directory), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink()

    def save_step(self, step, arrays, metadata=None):
        """Persist full NumPy arrays and metadata; return the NPZ path.

        Object arrays are rejected so all results load with
        ``np.load(path, allow_pickle=False)``. Failures propagate to the caller.
        """
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("Artifact step must be a nonnegative integer")
        step = int(step)
        for name, value in arrays.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
                    or name == "file"):
                raise ValueError("Invalid artifact array name: {!r}".format(name))
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError("Artifact array {!r} must be a NumPy array without object dtype".format(name))
        array_path = self.directory / "step_{:06d}.npz".format(step)
        metadata_path = array_path.with_suffix(".json")
        if array_path.exists() or metadata_path.exists():
            raise FileExistsError("Refusing to overwrite saved artifact step {} in {}".format(
                step, self.directory))
        document = _json_bytes({
            "schema_version": self.SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "run_id": self.run_id,
            "step": step,
            "metadata": metadata or {},
        })
        try:
            self._write_atomic(array_path, lambda handle: np.savez(handle, **arrays))
            self._write_atomic(metadata_path, lambda handle: handle.write(document))
        except Exception as exc:
            raise OSError("Cannot save artifact step {} in {}: {}".format(
                step, self.directory, exc)) from exc
        return array_path
