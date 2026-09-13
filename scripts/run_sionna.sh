#!/usr/bin/env bash
set -e
ATDF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${ATDF_SIONNA_PYTHON:-}" ]]; then
  ATDF_WORKER_PYTHON="$ATDF_SIONNA_PYTHON"
elif [[ -x "$ATDF_REPO_ROOT/.venv-sionna/bin/python" ]]; then
  ATDF_WORKER_PYTHON="$ATDF_REPO_ROOT/.venv-sionna/bin/python"
else
  ATDF_WORKER_PYTHON=python3
fi
exec "$ATDF_WORKER_PYTHON" "$ATDF_REPO_ROOT/sionna/atf_testing.py" \
  --scene "$ATDF_REPO_ROOT/assets/sionna/21202_building_full_room_shifted.xml" "$@"
