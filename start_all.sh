#!/usr/bin/env bash

# --- proxy sanitizer (avoid routing MJPG URL through local proxy) ---
# If you want to keep proxies for everything else, prefer the NO_PROXY allow-list below.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# Keep a safe NO_PROXY that includes local nets + your Pi
export NO_PROXY="localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12,$(jq -r '.pi_ip' "$(dirname "$0")/start_config.json")"
export no_proxy="$NO_PROXY"
echo "[start_all] Proxies disabled for this run. NO_PROXY=${NO_PROXY}"


CONFIG="$(dirname "$0")/start_config.json"

# --- helpers to read JSON ---
jqv () { jq -r "$1" "$CONFIG"; }

PI_USER="$(jqv '.pi_user')"
PI_IP="$(jqv '.pi_ip')"
HOST_IP="$(jqv '.host_ip')"

# Streaming settings
STREAM_TYPE="$(jqv '.stream_type // "mjpg"')"   # "gst" or "mjpg"
MJPG_URL="$(jqv '.mjpg_url // empty')"
GST_PORT="$(jqv '.gst_port // 5000')"

kill_pi_process() {
  echo "[start_all] Stopping Pi screen sessions…"
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${PI_USER}@${PI_IP}" '
      # screen -ls returns code 1 when no sessions; ignore errors
      sessions=$(screen -ls 2>/dev/null | sed -n "s/^[[:space:]]\([0-9]\+\.[^[:space:]]\+\)[[:space:]].*/\1/p" | grep -E "\.cam_session" || true)
      if [ -n "$sessions" ]; then
        for sid in $sessions; do
          # send Ctrl-C to that exact session id (e.g., 2848.cam_session)
          screen -S "$sid" -X stuff $'\003' || true
          sleep 0.5
          screen -S "$sid" -X quit || true
        done
      fi
      # belt & braces: ensure processes are gone
      pkill -f mjpg_streamer 2>/dev/null || true
      pkill -f "/Camera_Pi/LED.py" 2>/dev/null || true
      true
    ' || true
}

kill_host_process() {
  echo "[start_all] Stopping host launcher…"
  [[ -n "${HOST_LAUNCH_PID:-}" ]] && kill "${HOST_LAUNCH_PID}" 2>/dev/null || true
}

cleanup() {
  # Run both, always
  kill_host_process
  kill_pi_process
}

# trap on multiple signals (Ctrl+C, terminal close, kill, etc.)
trap cleanup INT TERM HUP EXIT

echo "[start_all] Using config:"
echo "  Pi: ${PI_USER}@${PI_IP}"
echo "  Host IP: ${HOST_IP}"
echo "  Stream: ${STREAM_TYPE}, MJPG_URL=${MJPG_URL:-N/A}, GST_PORT=${GST_PORT}"

# 1) Ensure screen on Pi and push config
echo "[start_all] Ensuring Pi has screen and pushing config…"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${PI_USER}@${PI_IP}" 'command -v screen >/dev/null || (sudo apt-get update && sudo apt-get install -y screen)'

scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$CONFIG" "${PI_USER}@${PI_IP}:/home/${PI_USER}/Camera_Pi/start_config_pi.json"

# 2) Pre-clean any stale sessions, then start a fresh one named "cam_session"
echo "[start_all] Cleaning old sessions on Pi…"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${PI_USER}@${PI_IP}" '
    sessions=$(screen -ls 2>/dev/null | sed -n "s/^[[:space:]]\([0-9]\+\.[^[:space:]]\+\)[[:space:]].*/\1/p" | grep -E "\.cam_session" || true)
    for sid in $sessions; do screen -S "$sid" -X quit || true; done
    pkill -f mjpg_streamer 2>/dev/null || true
    pkill -f "/Camera_Pi/LED.py" 2>/dev/null || true
    true
  '

echo "[start_all] Starting camera streamer on Pi in screen 'cam_session'…"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${PI_USER}@${PI_IP}" \
  "screen -dmS cam_session bash -lc 'cd ~/Camera_Pi && exec bash ./start_pi.sh &> ~/cam_session.log'"

# Verify screen session
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${PI_USER}@${PI_IP}" 'screen -list || true'

# 3) Start the host ROS 2 launch
echo "[start_all] Starting host ROS 2 launch…"
"$(dirname "$0")/start_host.sh" \
  "stream_type:=${STREAM_TYPE}" \
  "mjpg_url:=${MJPG_URL}" \
  "gst_port:=${GST_PORT}" &
HOST_LAUNCH_PID=$!

# Wait specifically on the host launch; signals will trigger the trap
wait "$HOST_LAUNCH_PID"
