#!/usr/bin/env bash

# --- 0) Proxy sanitizer (host side) ---
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12}"
export no_proxy="$NO_PROXY"
echo "[start_host] Proxies disabled for this run. NO_PROXY=${NO_PROXY}"

set -e

# --- 1) Load ROS 2 + venv + paths (defines PROJECT_ROOT) ---
source "$(dirname "$0")/env_loader.sh"

# --- 2) Overlay your workspace if it exists ---
WS="${PROJECT_ROOT}/ros2_ws"
if [[ -f "${WS}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${WS}/install/setup.bash"
else
  echo "[start_host] NOTE: ${WS}/install/setup.bash not found; launching with underlay only."
fi

echo "PYTHONPATH: ${PYTHONPATH}"

# --- 3) Optional GUI tools (non-blocking) ---
(rqt_graph >/dev/null 2>&1 &)
(rqt >/dev/null 2>&1 &)

# Optional: allow ROS_DOMAIN_ID override via env or arg, example:
# export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# --- 4) Launch the demo ---
exec ros2 launch gelwedge_demo_ros2 demo.launch.py "$@"
