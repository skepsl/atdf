#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/ros_env.bash"
ATDF_RECORD_PREFIX="${1:-$HOME/atdf_recordings/tro_run}"
mkdir -p -- "$(dirname -- "$ATDF_RECORD_PREFIX")"
exec roslaunch atdf_video record.launch output_prefix:="$ATDF_RECORD_PREFIX"
