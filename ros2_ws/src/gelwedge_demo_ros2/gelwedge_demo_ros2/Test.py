#!/usr/bin/env python3
from __future__ import annotations

import os
import copy
import cv2
import numpy as np
from pathlib import Path
import importlib.resources as ir  # to load packaged files

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, WrenchStamped

# Local package imports (the .so and helpers live in this same module dir)
from . import utilities_demo         # Utilities
from . import find_marker            # Compiled extension .so
from . import setting                # Params/defaults


def _load_transformation_matrix(logger) -> np.ndarray | None:
    """
    Load transformation_matrix.npy packaged with the module.
    """
    try:
        with ir.as_file(ir.files(__package__) / 'transformation_matrix.npy') as p:
            if not p.exists():
                logger.warn(f"transformation_matrix.npy not found at {p}")
                return None
            return np.load(str(p))
    except Exception as e:
        logger.error(f"Failed to load transformation_matrix.npy: {e}")
        return None

def _compose_transform(M_base, width=800, height=600):
    if M_base is None:
        return None, (width, height)

    R = np.array([[0, 1, 0],
                  [-1, 0, height],
                  [0, 0, 1]], dtype=np.float32)
    S = np.array([[0.5, 0,   0],
                  [0,   0.5, 0],
                  [0,   0,   1]], dtype=np.float32)

    M_final = S @ R @ M_base
    # Final size after rotate(→600x800) then scale(×0.5): (300, 400)
    out_size = (height // 2, width // 2)   # <-- note the swap
    return M_final, out_size


import threading

# class LatestFrame:
#     def __init__(self, src):
#         self.cap = cv2.VideoCapture(src)
#         self.lock = threading.Lock()
#         self.frame = None
#         self.alive = True
#         t = threading.Thread(target=self._reader, daemon=True)
#         t.start()
#
#     def isOpened(self):
#         return self.cap.isOpened()
#
#     def _reader(self):
#         while self.alive:
#             ret, f = self.cap.read()
#             if not ret:
#                 continue
#             with self.lock:
#                 self.frame = f
#
#     def read(self):
#         with self.lock:
#             return (self.frame is not None), self.frame
#
#     def set(self, prop_id, value):
#         return self.cap.set(prop_id, value)
#
#     def release(self):
#         self.alive = False
#         self.cap.release()
#

class GelWedgeDemoNode(Node):
    def __init__(self) -> None:
        super().__init__('gelwedge_demo')

        # --- Parameters (ROS 2 params) ---
        self.declare_parameter('camera_type', 'gs')   # 'mjpg' or 'gs'
        self.declare_parameter('mjpg_url', 'http://172.17.167.138:8080/?action=stream')
        self.declare_parameter('gst_port', 5000)
        self.declare_parameter('frame_width', 800)
        self.declare_parameter('frame_height', 600)

        # Flow→Wrench model params (tune these!)
        self.declare_parameter('pixel_size_m', getattr(setting, 'pixel_size_m', 5e-4))     # m/px
        self.declare_parameter('k_force_per_m', getattr(setting, 'k_force_per_m', 1.0))    # N per meter

        self.camera_type = self.get_parameter('camera_type').value
        self.mjpg_url = self.get_parameter('mjpg_url').value
        self.gst_port = int(self.get_parameter('gst_port').value)
        self.width = int(self.get_parameter('frame_width').value)
        self.height = int(self.get_parameter('frame_height').value)
        self.pixel_size_m = float(self.get_parameter('pixel_size_m').value)
        self.k_force_per_m = float(self.get_parameter('k_force_per_m').value)

        # ---------- Publishers ----------
        # legacy point (unused by default; keep for compatibility)
        self.pub_legacy_pt = self.create_publisher(Point, '/tactile', 10)
        # wrench (force/torque) and CoP
        self.pub_wrench = self.create_publisher(WrenchStamped, '/tactile/flow_wrench', 10)
        self.pub_cop    = self.create_publisher(PointStamped,   '/tactile/cop',        10)

        # ---------- Resources ----------
        setting.init()
        self.matcher = find_marker.Matching(
            N_=setting.N_, M_=setting.M_, fps_=setting.fps_,
            x0_=setting.x0_, y0_=setting.y0_, dx_=setting.dx_, dy_=setting.dy_,
        )
        self.M = _load_transformation_matrix(self.get_logger())

        self.M_comp, self.out_size = _compose_transform(self.M, self.width, self.height)

        # ---------- Video capture ----------
        if self.camera_type == 'mjpg':
            self.get_logger().info(f"Opening MJPG stream (latest-frame reader): {self.mjpg_url}")
            self.cap = LatestFrame(self.mjpg_url)
        else:  # 'gs'
            self.cap = self._open_capture(self.camera_type)

        # small buffer on backends that support it (no-op for LatestFrame)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap or not self.cap.isOpened():
            self.get_logger().error("Failed to open camera stream.")
            raise RuntimeError("Camera open failed")

        self.count = 0
        self.frame0 = None
        self.frame0_final = None

        # Run at camera rate (timer 0.0 spins once per available frame)
        self.timer = self.create_timer(0.033, self._loop_once)
        self.get_logger().info("GelWedgeDemoNode started.")

    # ---- helpers ----

    def _open_capture(self, camera_type: str):
        if camera_type == 'mjpg':
            self.get_logger().info(f"Opening MJPG stream: {self.mjpg_url}")
            cap = cv2.VideoCapture(self.mjpg_url)
        else:
            gst_pipeline = (
                f"udpsrc port={self.gst_port} ! application/x-rtp, encoding-name=H264 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            )
            self.get_logger().info(f"Opening GStreamer pipeline: {gst_pipeline}")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            self.get_logger().error("Failed to open camera stream.")
        return cap

    def _publish_legacy_point(self, x: float, y: float, z: float):
        msg = Point()
        msg.x = float(x); msg.y = float(y); msg.z = float(z)
        self.pub_legacy_pt.publish(msg)

    def _publish_wrench(self, Fx: float, Fy: float, tau_z: float, frame_id: str = 'map'):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.wrench.force.x = float(Fx)
        msg.wrench.force.y = float(Fy)
        msg.wrench.force.z = 0.0
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = float(tau_z)
        self.pub_wrench.publish(msg)

    def _publish_cop(self, cop_x_px: float, cop_y_px: float, frame_id: str = 'map'):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        # Publish in pixels (image coords). Convert to meters if you prefer:
        msg.point.x = float(cop_x_px)
        msg.point.y = float(cop_y_px)
        msg.point.z = 0.0
        self.pub_cop.publish(msg)

    # ---- main processing loop ----
    def _loop_once(self) -> None:
        ret, frame = self.cap.read()
        self.count += 1
        if not ret or frame is None:
            return

        # One-pass warp to final processed size/orientation
        if self.M_comp is not None:
            try:
                frame_proc = cv2.warpPerspective(frame, self.M_comp, self.out_size)
            except Exception as e:
                self.get_logger().warn(f"warpPerspective (fused) failed: {e}")
                frame_proc = utilities_demo.get_processed_frame(frame)  # fallback
        else:
            frame_proc = utilities_demo.get_processed_frame(frame)
        if frame_proc is None:
            return

        proc_w, proc_h = frame_proc.shape[1], frame_proc.shape[0]
        origin_px = (proc_w / 2.0, proc_h / 2.0)

        # if self.count == 1:
        #     self.frame0 = copy.deepcopy(frame_proc)
        #     self.frame0_final = utilities_demo.inpaint(self.frame0)
        #
        # frame_final = utilities_demo.inpaint(frame_proc)

        # contact_area_dilated = utilities_demo.difference(
        #     frame_final, self.frame0_final, debug=False
        # )
        # contours = utilities_demo.get_all_contour(contact_area_dilated, frame_proc, debug=False)
        # hull_area, hull_mask, slope, center = utilities_demo.get_convex_hull_area(
        #     contact_area_dilated, frame_proc, debug=False
        # )

        # Marker tracking
        m_centers = utilities_demo.marker_center_fast(frame_proc, debug=False)
        self.matcher.init(m_centers)
        self.matcher.run()
        flow = self.matcher.get_flow()

        frame_flow = utilities_demo.draw_flow(frame_proc, flow)
        # frame_flow_hull, avg_flow = utilities_demo.draw_flow_mask(
        #     frame_proc, flow, hull_mask, debug=False
        # )

        try:
            wrench = utilities_demo.compute_resultant_flow_wrench(
                flow,
                pixel_size_m=self.pixel_size_m,
                k_force_per_m=self.k_force_per_m,
                origin_px=origin_px,
                use_only_contact=True,
                weight_by_magnitude=True,
            )

            Fx, Fy, tau_z = 5*wrench['Fx'], -5*wrench['Fy'], 50*wrench['tau_z']
            cop_x, cop_y = wrench['cop_x'], wrench['cop_y']

            # Publish ROS 2 topics
            self._publish_wrench(Fx, Fy, tau_z, frame_id='map')
            if np.isfinite(cop_x) and np.isfinite(cop_y):
                self._publish_cop(cop_x, cop_y, frame_id='map')

            # Optional: overlay quick text
            vis = frame_flow
            cv2.putText(vis, f"Fx={Fx:.3f} N  Fy={Fy:.3f} N",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"tau_z={tau_z:.4f} Nmm",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"CoP(px)=({cop_x:.1f}, {cop_y:.1f})  n={wrench['n_used']}",
                        (10, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except AttributeError as e:
            # compute_resultant_flow_wrench missing
            self.get_logger().error(
                "utilities_demo.compute_resultant_flow_wrench not found. "
                "Please add it to utilities_demo.py."
            )
            vis = frame_flow
        except Exception as e:
            self.get_logger().warn(f"wrench computation failed: {e}")
            vis = frame_flow

        # Show (optional – safe in ROS 2; Ctrl+C stops node)
        if (self.count % 2) == 0:  # ~15 fps UI
            try:
                cv2.imshow("frame_flow", frame_flow)
                cv2.waitKey(1)
            except Exception:
                pass

        # # Publish
        # if slope is not None and avg_flow is not None:
        #     self._publish_point(avg_flow[0], avg_flow[1], slope)
        # else:
        #     self._publish_point(1.0, 2.0, 0.0)

    # ---- shutdown ----
    def destroy_node(self) -> bool:
        try:
            if hasattr(self, 'cap') and self.cap:
                # Works for both LatestFrame and VideoCapture
                if hasattr(self.cap, "release"):
                    self.cap.release()
            cv2.destroyAllWindows()
        finally:
            return super().destroy_node()


def main():
    rclpy.init()
    node = GelWedgeDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
