#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/ros_env.bash"
exec roslaunch atdf_video navigator.launch "$@"
