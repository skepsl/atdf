# Sourced by the ROS launch scripts; never source this in the Sionna environment.
ATDF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f "$ATDF_REPO_ROOT/.venv-ros/bin/activate" ]]; then
  source "$ATDF_REPO_ROOT/.venv-ros/bin/activate"
fi
if [[ ! -f "$ATDF_REPO_ROOT/devel/setup.bash" ]]; then
  echo "Build this checkout first: $ATDF_REPO_ROOT/scripts/build.sh" >&2
  return 1
fi
source "$ATDF_REPO_ROOT/devel/setup.bash"
cd "$ATDF_REPO_ROOT"
