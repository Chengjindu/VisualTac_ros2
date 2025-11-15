#!/usr/bin/env bash
set -e

# 1) Load ROS 2 + venv + paths (this defines PROJECT_ROOT)
source "$(dirname "$0")/env_loader.sh"

# 2) Overlay your workspace (relative to the project root)
WS="${PROJECT_ROOT}/ros2_ws"
if [[ -f "${WS}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${WS}/install/setup.bash"
else
  echo "[calibrate] WARN: ${WS}/install/setup.bash not found. Did you build? (colcon build)"
fi

echo "PYTHONPATH: ${PYTHONPATH:-<empty>}"

# 3) Optional GUI tools (non-blocking)
(rqt_graph >/dev/null 2>&1 &)
(rqt >/dev/null 2>&1 &)

# 4) Launch calibration (passes through any extra args)
exec ros2 launch gelwedge_demo_ros2 calibrate.launch.py "$@"
