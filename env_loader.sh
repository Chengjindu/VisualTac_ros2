#!/usr/bin/env bash
# remove `-u` to avoid "unbound variable" inside ROS setup scripts
#set -e -o pipefail

# --- figure out project root from this file's location ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"   # default = folder containing this script
VENV_PATH="$PROJECT_ROOT/.venv"

# ROS underlay
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS_BIN="/opt/ros/humble/bin"
ROS_PY="/opt/ros/humble/lib/python3.10/site-packages"

# --- source ROS 2 underlay ---
if [[ -f "$ROS_SETUP" ]]; then
  source "$ROS_SETUP"
else
  echo "ERROR: $ROS_SETUP not found (is ROS 2 Humble installed?)"
  exit 1
fi

# --- activate venv ---
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
else
  echo "WARN: venv not found at $VENV_PATH (continuing without it)"
fi

# --- ensure ROS site-packages in PYTHONPATH (inside venv) ---
if [[ -d "$ROS_PY" && ":${PYTHONPATH-}:" != *":$ROS_PY:"* ]]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROS_PY"
fi

# (optional) add your workspace site-packages if already built
WS_PY="$PROJECT_ROOT/ros2_ws/install/gelwedge_demo_ros2/lib/python3.10/site-packages"
if [[ -d "$WS_PY" && ":${PYTHONPATH-}:" != *":$WS_PY:"* ]]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$WS_PY"
fi

# --- ensure ROS CLI on PATH ---
if [[ ":$PATH:" != *":$ROS_BIN:"* ]]; then
  export PATH="$ROS_BIN:$PATH"
fi

echo "Environment ready."
echo "  Project root: $PROJECT_ROOT"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
echo "  Python=$(python -V)"
echo "  PATH includes: $ROS_BIN"
echo "  PYTHONPATH=$PYTHONPATH"
