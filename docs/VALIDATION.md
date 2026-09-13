# Packaging validation

Checked on 2026-09-13. These checks cover the published copy, its saved example,
and startup dependencies. No new Isaac Sim/Sionna experiment was run.

## Completed checks

- **99 automated tests passed:** 76 in `tests/` and 23 in
  `src/atdf_video/tests/`. These cover the finder helpers, navigation timing,
  measurement geometry, saved plots/snapshots, and the Sionna file protocol.
- A clean catkin CMake configure, build, and install succeeded for both ROS
  packages. Build, devel, and install outputs were kept outside this repository.
- `roslaunch --dump-params atdf_video example_8985.launch` resolved the copied
  configuration, checkpoint, and map. The example's initial pose, source
  position, particle bounds, Sionna measurement source, and enabled PDF saving
  were checked without launching nodes.
- A CPU smoke check, run from outside the repository, loaded the default
  Neural Ray Predictor checkpoint (12,826,375 parameters) and initialized
  1,000 finite particles inside the configured source bounds.
- The standalone worker's `--help` works without third-party packages. Protocol
  tests exercise fresh and idle exchange directories, restart handling, atomic
  response/acknowledgement writes, and a request through the real copied client
  with a simulated ray response. They do not execute Sionna ray tracing.
- During packaging, the worker's numerical conversion was compared with the
  original reference implementation for zero, five, and 25 rays across six
  transmitter/receiver pairs. Arrays and ray-solver arguments matched. This
  comparison used synthetic inputs, not a new simulator experiment.
- Shell script syntax and local documentation links were checked. Asset
  references and hashes were checked as described in [ASSETS.md](ASSETS.md).
- The example PNGs and GIF were rendered from the original saved PDFs. Original
  PDFs, numerical snapshots, and per-step JSON were preserved byte for byte.
  The example manifest records the two portable path substitutions in its
  top-level metadata. GIF timing is for presentation, not simulation playback.

## Validation environment

The ROS-side checks used the existing Ubuntu 20.04/ROS Noetic environment:
Python 3.8.10, PyTorch 2.4.1+cu118, NumPy 1.17.4, SciPy 1.10.1,
Matplotlib 3.1.2, Pillow 7.0.0, and PyYAML 5.3.1.

That existing environment emits a SciPy warning because its NumPy is older
than SciPy supports. The supplied `requirements-ros.txt` instead selects
Python 3.8-compatible versions including NumPy 1.24.4. A clean installation
of all supplied dependency pins has **not** been tested. The successful checks
above therefore do not establish a tested dependency lockfile.

The installed Sionna environment's package versions were inspected and are
listed in [ASSETS.md](ASSETS.md). It was not used for a new end-to-end trial.

## Remaining runtime checks

The bundled Isaac stage is a candidate with the expected ROS1 interfaces;
it has not been established as the exact stage used for run 8985. Three
upstream NVIDIA material URLs remain, and may require network access or an
existing cache. See [ASSETS.md](ASSETS.md) for the exact dependencies.

Fresh ROS navigation, Isaac motion, Sionna ray tracing, and their complete
interaction still need a trial on the intended simulation machine. The
recorded example demonstrates the saved historical run, not a rerun of this
portable copy. Its last estimate is accurate to approximately 0.0693 m, but
the final saved frame does not establish that the convergence criterion was
met.

## Repeat the automated checks

From the repository root after following the installation instructions:

```bash
source /opt/ros/noetic/setup.bash
source .venv-ros/bin/activate
python -m unittest discover -s tests -v
python -m unittest discover -s src/atdf_video/tests -v
python sionna/atf_testing.py --help
./scripts/build.sh
```

The scripts and the [README](../README.md) provide the commands for the
subsequent simulation trial.
