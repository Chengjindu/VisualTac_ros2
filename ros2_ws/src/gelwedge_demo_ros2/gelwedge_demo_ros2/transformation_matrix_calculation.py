#!/usr/bin/env python3
"""
transformation_matrix_calculation.py
------------------------------------
Interactive utility to select 4 reference points from a camera frame
and compute the perspective transform matrix used by the GelWedge demo.

Examples:
  # Use values from start_config.json (auto-discovered)
  python3 -m gelwedge_demo_ros2.transformation_matrix_calculation

  # Explicit config path
  python3 -m gelwedge_demo_ros2.transformation_matrix_calculation --config ~/Projects/Project_VisualTac/start_config.json

  # Override source/URL/port
  python3 -m gelwedge_demo_ros2.transformation_matrix_calculation --source mjpg --url http://<pi>:8080/?action=stream
  python3 -m gelwedge_demo_ros2.transformation_matrix_calculation --source gs --port 5000
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------- util: config discovery ----------------
def _candidate_configs(here: Path) -> list[Path]:
    """
    Return an ordered list of candidate paths where start_config.json might live.
    Updated for new project layout at ~/Projects/Project_VisualTac.
    """
    # Common anchors
    home = Path.home()

    roots = [
        home / "Projects" / "Project_VisualTac",
        home / "Project_VisualTac",  # legacy location (old path)
    ]

    # Walk up a few levels from the module file and try each level
    ups = [here.parent, *list(here.parents)[1:5]]

    cands: list[Path] = []

    # 1) Current working dir
    cands.append(Path.cwd() / "start_config.json")

    # 2) From module upwards (…/start_config.json)
    for d in ups:
        cands.append(d / "start_config.json")
        # If we see a ros2_ws, try its parent as project root too
        if "ros2_ws" in d.parts:
            try:
                idx = d.parts.index("ros2_ws")
                proj = Path(*d.parts[:idx])
                if proj:
                    cands.append(proj / "start_config.json")
            except ValueError:
                pass

    # 3) Explicit known roots and their ros2_ws siblings
    for r in roots:
        cands.append(r / "start_config.json")
        cands.append(r / "ros2_ws" / "start_config.json")

    # Deduplicate while preserving order
    out, seen = [], set()
    for p in cands:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _load_start_config(path: Optional[str]) -> tuple[dict, Optional[Path]]:
    """
    Load the JSON dict from explicit path, env START_CONFIG, or common locations.
    Returns ({}, None) if nothing found/readable.
    """
    # 1) explicit arg
    if path:
        p = Path(path).expanduser().resolve()
        try:
            return json.loads(p.read_text()), p
        except Exception:
            return {}, None

    # 2) env var
    env_p = os.environ.get("START_CONFIG")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        try:
            return json.loads(p.read_text()), p
        except Exception:
            return {}, None

    # 3) common places
    here = Path(__file__).resolve()
    for p in _candidate_configs(here):
        try:
            if p.exists():
                return json.loads(p.read_text()), p
        except Exception:
            pass

    return {}, None


# ---------------- video helpers ----------------
def _open_capture(source_type: str, port: int, url: str):
    """
    Open the appropriate video stream depending on source_type.
    """
    if source_type == "mjpg":
        cap = cv2.VideoCapture(url)
    elif source_type == "gs":
        gst_pipeline = (
            f"udpsrc port={port} ! application/x-rtp, encoding-name=H264 ! "
            "rtph264depay ! h264parse ! queue ! avdec_h264 ! queue ! videoconvert ! queue ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video capture ({source_type})")
    return cap


def _select_points(frame) -> np.ndarray:
    """
    Let the user click 4 points on the given frame.
    Returns: float32 array of shape (4, 2)
    """
    src_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            src_points.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select 4 Points', frame)
            if len(src_points) == 4:
                print("✅ 4 points selected. Press any key to continue...")

    cv2.imshow('Select 4 Points', frame)
    cv2.setMouseCallback('Select 4 Points', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.float32(src_points)


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Compute perspective transform for GelWedge Demo")
    parser.add_argument("--config", help="Path to start_config.json (optional)")
    parser.add_argument("--source", choices=["mjpg", "gs"], help="Override video source type")
    parser.add_argument("--url", help="Override MJPG stream URL")
    parser.add_argument("--port", type=int, help="Override UDP port for GStreamer input")
    parser.add_argument("--width", type=int, default=800, help="Output frame width")
    parser.add_argument("--height", type=int, default=600, help="Output frame height")
    args = parser.parse_args()

    # Load config and compute defaults
    cfg, cfg_path = _load_start_config(args.config)
    cfg_source = (cfg.get("stream_type") or "mjpg").lower()
    cfg_url = cfg.get("mjpg_url") or "http://127.0.0.1:8080/?action=stream"
    cfg_port = int(cfg.get("gst_port") or 5000)

    # CLI overrides config
    source = (args.source or cfg_source).lower()
    url = args.url or cfg_url
    port = int(args.port) if args.port is not None else cfg_port

    print("=== transformation_matrix_calculation ===")
    print(f"Config file: {cfg_path if cfg_path else '(auto-discovery failed → using defaults/overrides)'}")
    print(f"Using source={source} url={url if source=='mjpg' else '-'} port={port if source=='gs' else '-'}")

    # --- Open video ---
    try:
        cap = _open_capture(source, port, url)
        print(f"✅ Video capture opened ({source})")
    except Exception as e:
        print(f"❌ {e}")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to capture initial frame.")
        cap.release()
        return

    # --- Select points ---
    print("Click 4 corner points (top-left → top-right → bottom-right → bottom-left).")
    selected_points = _select_points(frame)
    if len(selected_points) != 4:
        print("❌ You must select exactly 4 points.")
        cap.release()
        return

    # --- Compute and save matrix ---
    dst_points = np.float32([
        [0, 0],
        [args.width, 0],
        [args.width, args.height],
        [0, args.height],
    ])
    M = cv2.getPerspectiveTransform(selected_points, dst_points)

    # Save next to this module
    save_path = Path(__file__).resolve().parent / "transformation_matrix.npy"
    np.save(str(save_path), M)
    print(f"✅ Transformation matrix saved to: {save_path}")

    cap.release()


if __name__ == "__main__":
    main()
