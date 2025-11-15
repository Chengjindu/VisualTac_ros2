#!/usr/bin/env bash
set -e

# --- 0) Env (defines PROJECT_ROOT, PATH, PYTHONPATH, venv, ROS, etc.) ---
#     Also keeps this script relocatable.
source "$(dirname "$0")/env_loader.sh"

CONFIG="$(dirname "$0")/start_config.json"

# --- 1) Read config ---
jqv () { jq -r "$1" "$CONFIG"; }
PI_USER="$(jqv '.pi_user')"
PI_IP="$(jqv '.pi_ip')"
HOST_IP="$(jqv '.host_ip')"
STREAM_TYPE="$(jqv '.stream_type // "mjpg"')"
MJPG_URL="$(jqv '.mjpg_url // empty')"
GST_PORT="$(jqv '.gst_port // 5000')"

ssh_pi() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"
}

# --- 2) Cleanup handlers (Pi + host) ---
kill_pi_processes() {
  echo "[start_calibration] Stopping Pi cam sessions & streamer…"
  ssh_pi "${PI_USER}@${PI_IP}" '
    # Quit all cam_session screens (use exact id)
    ids=$(screen -ls 2>/dev/null | sed -n "s/^[[:space:]]\([0-9]\+\.[^[:space:]]\+\)[[:space:]].*/\1/p" | grep -E "\.cam_session" || true)
    for sid in $ids; do
      screen -S "$sid" -X stuff $'\''\003'\'' || true   # Ctrl-C
      sleep 0.5
      screen -S "$sid" -X quit || true
    done
    # ensure processes are gone
    pkill -f mjpg_streamer 2>/dev/null || true
    pkill -f "/Camera_Pi/LED.py" 2>/dev/null || true
    true
  ' || true
}

kill_host_job() {
  echo "[start_calibration] Stopping host calibrate launcher…"
  [[ -n "${HOST_LAUNCH_PID:-}" ]] && kill "${HOST_LAUNCH_PID}" 2>/dev/null || true
}

cleanup() {
  kill_host_job
  kill_pi_processes
}

trap cleanup INT TERM HUP EXIT

echo "[start_calibration] Using config:"
echo "  Pi: ${PI_USER}@${PI_IP}"
echo "  Host IP: ${HOST_IP}"
echo "  Stream: ${STREAM_TYPE}, MJPG_URL=${MJPG_URL:-N/A}, GST_PORT=${GST_PORT}"

# --- 3) Prepare Pi & push config ---
echo "[start_calibration] Ensuring 'screen' on Pi and pushing config…"
ssh_pi "${PI_USER}@${PI_IP}" 'command -v screen >/dev/null || (sudo apt-get update && sudo apt-get install -y screen)'
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$CONFIG" "${PI_USER}@${PI_IP}:/home/${PI_USER}/Camera_Pi/start_config_pi.json"

# --- 4) Clean previous sessions/ports, then start a fresh streamer ---
echo "[start_calibration] Cleaning old sessions on Pi…"
kill_pi_processes

echo "[start_calibration] Starting camera streamer on Pi in screen 'cam_session'…"
ssh_pi "${PI_USER}@${PI_IP}" \
  "screen -dmS cam_session bash -lc 'cd ~/Camera_Pi && exec bash ./start_pi.sh &> ~/cam_session.log'"

# Show current screens (for visibility, not required)
ssh_pi "${PI_USER}@${PI_IP}" 'screen -list || true'

# --- 5) Wait for stream to become reachable (bypass proxies) ---
echo "[start_calibration] Waiting for stream to become ready…"
if [[ "$STREAM_TYPE" == "mjpg" ]]; then
  # curl without proxies; accept partial output; retry a few times
  for i in {1..15}; do
    if env -i curl --max-time 2 --silent --show-error --noproxy "*" "$MJPG_URL" | head -c 1 >/dev/null; then
      echo "[start_calibration] MJPG reachable."
      break
    fi
    sleep 0.6
    [[ $i -eq 15 ]] && { echo "[start_calibration] ERROR: MJPG not reachable."; exit 1; }
  done
else
  # For GStreamer/UDP: check port is open on the host side seeing packets
  for i in {1..15}; do
    if timeout 2 bash -lc "ss -u -lpn | grep -q \":${GST_PORT} \"" 2>/dev/null; then
      echo "[start_calibration] GST port appears active."
      break
    fi
    sleep 0.6
    [[ $i -eq 15 ]] && { echo "[start_calibration] WARNING: Could not confirm GST traffic; proceeding."; }
  done
fi

# --- 6) Start the host calibration launch (env already set by env_loader.sh) ---
echo "[start_calibration] Launching calibration UI…"
"$(dirname "$0")/calibrate_matrix.sh" \
  "stream_type:=${STREAM_TYPE}" \
  "mjpg_url:=${MJPG_URL}" \
  "gst_port:=${GST_PORT}" &
HOST_LAUNCH_PID=$!

# Wait specifically on calibration process; trap will clean up on exit
wait "$HOST_LAUNCH_PID"
