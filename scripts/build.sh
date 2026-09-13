#!/usr/bin/env bash
set -e
ATDF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f "$ATDF_REPO_ROOT/.venv-ros/bin/activate" ]]; then
  source "$ATDF_REPO_ROOT/.venv-ros/bin/activate"
fi
cd "$ATDF_REPO_ROOT"
exec catkin_make -DPYTHON_EXECUTABLE="$(command -v python3)" "$@"
