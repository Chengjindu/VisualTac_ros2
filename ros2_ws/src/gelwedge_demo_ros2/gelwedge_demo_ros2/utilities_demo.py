#!/usr/bin/env python3
"""
utilities_demo.py
-----------------
Collection of helper functions for GelWedge visual-tactile processing.

These functions handle image preprocessing, marker detection,
contact region analysis, optical flow visualization, and basic geometry.
They are generic OpenCV/Numpy utilities and fully compatible with ROS 2.
"""

import cv2
import numpy as np

# Prebuilt kernels so we don't allocate every frame
_K5  = np.ones((5, 5),  np.uint8)
_K11 = np.ones((11, 11), np.uint8)

from . import setting  # <-- relative import, uses global constants (N_, M_, dx_, …)


# ---------------------------------------------------------------------
#   FRAME PREPROCESSING
# ---------------------------------------------------------------------
def get_processed_frame(frame):
    """Resize, rotate and downsample incoming frame."""
    if frame is None or frame.size == 0:
        print("⚠️  Invalid frame received.")
        return None

    # Resize to 800×600 (expected size)
    if frame.shape[1] != 800 or frame.shape[0] != 600:
        frame = cv2.resize(frame, (800, 600))

    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    downsampled = cv2.pyrDown(rotated).astype(np.uint8)
    return downsampled

# def get_processed_frame(frame):
#     """Resize incoming frame to 800×600 (legacy fallback)."""
#     if frame is None or frame.size == 0:
#         print("⚠️  Invalid frame received.")
#         return None
#     if frame.shape[1] != 800 or frame.shape[0] != 600:
#         frame = cv2.resize(frame, (800, 600))
#     return frame

# ---------------------------------------------------------------------
#   MARKER SEGMENTATION / CENTERS
# ---------------------------------------------------------------------
# def mask_marker(frame, debug=False):
#     """Highlight bright marker dots in the image."""
#     m, n = frame.shape[1], frame.shape[0]
#     f32 = cv2.pyrDown(frame).astype(np.float32)
#
#     blur1 = cv2.GaussianBlur(f32, (25, 25), 0)
#     blur2 = cv2.GaussianBlur(f32, (15, 15), 0)
#     diff = blur1 - blur2
#     diff *= 20
#     diff = np.clip(diff, 0.0, 255.0)
#
#     TH = 120
#     mask_b = diff[:, :, 0] > TH
#     mask_g = diff[:, :, 1] > TH
#     mask_r = diff[:, :, 2] > TH
#     mask = (mask_b & mask_g) | (mask_b & mask_r) | (mask_g & mask_r)
#
#     if debug:
#         cv2.imshow("maskdiff", diff.astype(np.uint8))
#         cv2.imshow("mask", mask.astype(np.uint8) * 255)
#
#     mask = cv2.resize(mask.astype(np.uint8), (m, n))
#     return mask * 255

def mask_marker(frame, debug=False):
    """
    Highlight bright marker dots in the image using DoG (blur1 - blur2).
    If the frame is already small (~≤ 480 px on the long side), skip pyrDown
    and use smaller kernels for speed.
    """
    h, w = frame.shape[:2]
    small_input = max(h, w) <= 480

    # choose processing image and kernels
    if small_input:
        f32 = frame.astype(np.float32)
        k1, k2 = (15, 15), (9, 9)
    else:
        f32 = cv2.pyrDown(frame).astype(np.float32)
        k1, k2 = (25, 25), (15, 15)

    blur1 = cv2.GaussianBlur(f32, k1, 0)
    blur2 = cv2.GaussianBlur(f32, k2, 0)
    diff  = (blur1 - blur2) * 20.0
    diff  = np.clip(diff, 0.0, 255.0)

    TH = 120  # tune if needed after the kernel change
    # threshold any two channels above TH
    ch0 = diff[:, :, 0] > TH
    ch1 = diff[:, :, 1] > TH
    ch2 = diff[:, :, 2] > TH
    mask = (ch0 & ch1) | (ch0 & ch2) | (ch1 & ch2)
    mask = (mask.astype(np.uint8) * 255)

    # Resize mask back to original input size when we downscaled
    if not small_input:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if debug:
        cv2.imshow("maskdiff", diff.astype(np.uint8))
        cv2.imshow("mask", mask)

    return mask



def marker_center_fast(frame, debug=False):
    area_min, area_max = 20, 500
    mask = mask_marker(frame, debug)
    # mask is already uint8
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if not (area_min < a < area_max):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = max(w, h) / (min(w, h) + 1e-6)
        if ratio > 2.0:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        centers.append([m["m10"]/m["m00"], m["m01"]/m["m00"]])
    return centers



def marker_center(frame, debug=False):
    """Detect individual marker centroids."""
    area_min, area_max = 20, 500
    centers = []
    mask = mask_marker(frame, debug)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 5:
        print(f"⚠️  Too few markers: {len(contours)}")
        return centers

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        A = cv2.contourArea(contour)
        if area_min < A < area_max and abs(max(w, h) / min(w, h) - 1) < 1:
            M = cv2.moments(contour)
            centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    return centers


# ---------------------------------------------------------------------
#   CONTACT AREA ANALYSIS
# ---------------------------------------------------------------------
def inpaint(frame):
    """Remove bright markers from frame."""
    mask = mask_marker(frame)
    return cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)


# def difference(frame, frame0, debug=False):
#     """Compute binary contact area mask between current and reference."""
#     diff = (frame * 1.0 - frame0) / 255.0 + 0.5
#     diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
#     diff_uint8 = (diff * 255).astype(np.uint8)
#     diff_uint8[diff_uint8 > 140] = 255
#     diff_uint8[diff_uint8 <= 140] = 0
#
#     gray = cv2.cvtColor(diff_uint8, cv2.COLOR_BGR2GRAY)
#     _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
#     er = cv2.erode(th, np.ones((5, 5), np.uint8), iterations=2)
#     di = cv2.dilate(er, np.ones((5, 5), np.uint8), iterations=1)
#
#     if debug:
#         cv2.imshow("diff", gray)
#         cv2.imshow("mask", di)
#     return di

def difference(frame, frame0, debug=False):
    """Compute binary contact area mask between current and reference."""
    diff = (frame.astype(np.float32) - frame0.astype(np.float32)) / 255.0 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5

    diff_u8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff_u8, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    er = cv2.erode(th, _K5, iterations=2)
    di = cv2.dilate(er, _K5, iterations=1)

    if debug:
        cv2.imshow("diff", gray)
        cv2.imshow("mask", di)
    return di



def get_all_contour(diff_mask, frame, debug=False):
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        merged = np.concatenate(contours)
        ellipse = cv2.fitEllipse(merged)
        if debug:
            img = frame.copy()
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)
            cv2.imshow("Contact ellipse", img)
    except Exception:
        pass
    return contours


def regress_line(points, frame, debug=False):
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = vy / vx
    angle = -np.degrees(np.arctan(slope))
    lefty = int((-x * vy / vx) + y)
    righty = int(((frame.shape[1] - x) * vy / vx) + y)
    pt1, pt2 = (frame.shape[1] - 1, righty), (0, lefty)
    midx, midy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
    if debug:
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        cv2.circle(frame, (midx, midy), 10, (0, 0, 255), 2)
    return angle, (midy, midx)


def get_convex_hull_area(diff_mask, frame, debug=False):
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = frame.copy()
    hull_area, slope, center = 0, None, None
    hull_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
    if contours:
        try:
            pts = np.vstack(contours)[:, 0, :]
            hull_pts = cv2.convexHull(pts)
            hull_area = cv2.contourArea(hull_pts)
            slope, center = regress_line(pts, img)
            cv2.fillPoly(hull_mask, [hull_pts], 255)
        except Exception as e:
            print("⚠️  Hull error:", e)
    if debug:
        cv2.imshow("Convex Hull", img)
    return hull_area, hull_mask, slope, center


# ---------------------------------------------------------------------
#   FLOW VISUALIZATION
# ---------------------------------------------------------------------
def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occ = flow
    drawn = frame.copy()
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Cx[i][j]), int(Cy[i][j]))
            pt2 = (int(Cx[i][j] + (Cx[i][j] - Ox[i][j])),
                   int(Cy[i][j] + (Cy[i][j] - Oy[i][j])))
            color = (255, 255, 255) if Occ[i][j] <= -1 else (0, 255, 255)
            cv2.arrowedLine(drawn, pt1, pt2, color, 2, tipLength=0.2)
    return drawn

def draw_flow_fast(frame, flow, K=1, step=4):
    Ox, Oy, Cx, Cy, Occupied = [np.asarray(a)[::step, ::step] for a in flow]
    start = np.stack([Cx, Cy], axis=-1).astype(np.int32)
    end = (start + K * np.stack([Cx - Ox, Cy - Oy], axis=-1)).astype(np.int32)
    drawn = frame.copy()

    for color, mask in [((0,255,255), Occupied > -1), ((255,255,255), Occupied <= -1)]:
        pts = np.column_stack((start[mask], end[mask]))
        if len(pts):
            cv2.polylines(drawn, pts.reshape(-1, 2, 2), False, color, 2)
    return drawn


def draw_flow_mask(frame, flow, mask, debug=False):
    Ox, Oy, Cx, Cy, Occ = flow
    K = 2
    drawn = frame.copy()
    mask_d = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    drawn_mask = cv2.bitwise_and(drawn, drawn, mask=mask)
    change = np.zeros(2, np.float32)
    counter = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            cx, cy = int(Cx[i][j]), int(Cy[i][j])
            if 0 <= cy < mask_d.shape[0] and 0 <= cx < mask_d.shape[1]:
                if mask_d[cy, cx] == 255:
                    dx, dy = cx - int(Ox[i][j]), cy - int(Oy[i][j])
                    pt1, pt2 = (cx, cy), (int(cx + K*dx), int(cy + K*dy))
                    counter += 1
                    change += [dx, dy]
                    color = (255, 255, 255) if Occ[i][j] <= -1 else (0, 255, 255)
                    cv2.arrowedLine(drawn_mask, pt1, pt2, color, 2, tipLength=0.2)
    if counter > 0:
        change /= counter
        cv2.putText(drawn_mask, f"Average: {change}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if debug:
        cv2.imshow("Flow Hull", drawn_mask)
    return drawn_mask, change.tolist()

def compute_resultant_flow_wrench(
    flow,
    pixel_size_m=5e-4,        # meters per pixel
    k_force_per_m=1.0,        # N per meter (calib gain)
    origin_px=None,           # (ox, oy) in pixels; default image center
    use_only_contact=True,
    weight_by_magnitude=True, # uses L1 magnitude (|dx|+|dy|), no sqrt
):
    Ox, Oy, Cx, Cy, Occ = flow

    # Pack to small ints to reduce memory bandwidth; keep sums in larger dtype
    Ox = np.asarray(Ox, dtype=np.int16); Oy = np.asarray(Oy, dtype=np.int16)
    Cx = np.asarray(Cx, dtype=np.int16); Cy = np.asarray(Cy, dtype=np.int16)
    Occ = np.asarray(Occ, dtype=np.int8)

    # Displacements in pixels (int16)
    dx = (Cx - Ox).astype(np.int16)
    dy = (Cy - Oy).astype(np.int16)

    mask = (Occ > -1) if use_only_contact else np.ones_like(Occ, dtype=bool)
    if not np.any(mask):
        return dict(Fx=0.0, Fy=0.0, tau_z=0.0,
                    cop_x=np.nan, cop_y=np.nan,
                    n_used=0, avg_disp_px=0.0)

    dx_m = dx[mask].astype(np.int16)
    dy_m = dy[mask].astype(np.int16)
    Px   = Cx[mask].astype(np.int16)
    Py   = Cy[mask].astype(np.int16)

    # Force accumulation in pixel units (int64 to be safe)
    sum_dx = np.sum(dx_m, dtype=np.int64)
    sum_dy = np.sum(dy_m, dtype=np.int64)

    # Torque in pixel^2 units:
    # tau_px2 = sum( (Px-ox)*dy - (Py-oy)*dx )
    if origin_px is None:
        h, w = Occ.shape[:2]
        ox, oy = (w // 2), (h // 2)
    else:
        ox, oy = origin_px
        ox = int(ox); oy = int(oy)

    rx = (Px.astype(np.int32) - ox)  # int32 to avoid int16 overflow
    ry = (Py.astype(np.int32) - oy)
    tau_px2 = np.sum(rx * dy_m.astype(np.int32) - ry * dx_m.astype(np.int32), dtype=np.int64)

    # CoP (pixel) with L1 weights—no sqrt
    if weight_by_magnitude:
        wgt = (np.abs(dx_m).astype(np.int32) + np.abs(dy_m).astype(np.int32))
        ws  = int(np.sum(wgt, dtype=np.int64))
        if ws > 0:
            cop_x = float(np.sum(Px.astype(np.int32) * wgt, dtype=np.int64) / ws)
            cop_y = float(np.sum(Py.astype(np.int32) * wgt, dtype=np.int64) / ws)
        else:
            cop_x = float(np.mean(Px)); cop_y = float(np.mean(Py))
        avg_disp_px = float(np.mean(wgt) * 0.5)  # ~L1/2 ≈ L2 magnitude
    else:
        cop_x = float(np.mean(Px)); cop_y = float(np.mean(Py))
        avg_disp_px = float(0.5 * (np.mean(np.abs(dx_m)) + np.mean(np.abs(dy_m))))

    # Single scaling step to SI:
    s1 = float(k_force_per_m * pixel_size_m)            # N per pixel
    s2 = float(k_force_per_m * (pixel_size_m**2))       # N·m per pixel^2

    Fx = s1 * float(sum_dx)
    Fy = s1 * float(sum_dy)
    tau_z = s2 * float(tau_px2)

    return dict(
        Fx=Fx, Fy=Fy, tau_z=tau_z,
        cop_x=cop_x, cop_y=cop_y,
        n_used=int(dx_m.size),
        avg_disp_px=avg_disp_px,
    )

def compute_resultant_flow_wrench_int(
    flow,
    pixel_size_m=5e-4,        # meters per pixel
    k_force_per_m=1.0,        # N per meter (calib gain)
    origin_px=None,           # (ox, oy) in pixels; default image center
    use_only_contact=True,
    weight_by_magnitude=True, # uses L1 magnitude (|dx|+|dy|), no sqrt
):
    Ox, Oy, Cx, Cy, Occ = flow

    # Pack to small ints to reduce memory bandwidth; keep sums in larger dtype
    Ox = np.asarray(Ox, dtype=np.int16); Oy = np.asarray(Oy, dtype=np.int16)
    Cx = np.asarray(Cx, dtype=np.int16); Cy = np.asarray(Cy, dtype=np.int16)
    Occ = np.asarray(Occ, dtype=np.int8)

    # Displacements in pixels (int16)
    dx = (Cx - Ox).astype(np.int16)
    dy = (Cy - Oy).astype(np.int16)

    mask = (Occ > -1) if use_only_contact else np.ones_like(Occ, dtype=bool)
    if not np.any(mask):
        return dict(Fx=0.0, Fy=0.0, tau_z=0.0,
                    cop_x=np.nan, cop_y=np.nan,
                    n_used=0, avg_disp_px=0.0)

    dx_m = dx[mask].astype(np.int16)
    dy_m = dy[mask].astype(np.int16)
    Px   = Cx[mask].astype(np.int16)
    Py   = Cy[mask].astype(np.int16)

    # Force accumulation in pixel units (int64 to be safe)
    sum_dx = np.sum(dx_m, dtype=np.int64)
    sum_dy = np.sum(dy_m, dtype=np.int64)

    # Torque in pixel^2 units:
    # tau_px2 = sum( (Px-ox)*dy - (Py-oy)*dx )
    if origin_px is None:
        h, w = Occ.shape[:2]
        ox, oy = (w // 2), (h // 2)
    else:
        ox, oy = origin_px
        ox = int(ox); oy = int(oy)

    rx = (Px.astype(np.int32) - ox)  # int32 to avoid int16 overflow
    ry = (Py.astype(np.int32) - oy)
    tau_px2 = np.sum(rx * dy_m.astype(np.int32) - ry * dx_m.astype(np.int32), dtype=np.int64)

    # CoP (pixel) with L1 weights—no sqrt
    if weight_by_magnitude:
        wgt = (np.abs(dx_m).astype(np.int32) + np.abs(dy_m).astype(np.int32))
        ws  = int(np.sum(wgt, dtype=np.int64))
        if ws > 0:
            cop_x = float(np.sum(Px.astype(np.int32) * wgt, dtype=np.int64) / ws)
            cop_y = float(np.sum(Py.astype(np.int32) * wgt, dtype=np.int64) / ws)
        else:
            cop_x = float(np.mean(Px)); cop_y = float(np.mean(Py))
        avg_disp_px = float(np.mean(wgt) * 0.5)  # ~L1/2 ≈ L2 magnitude
    else:
        cop_x = float(np.mean(Px)); cop_y = float(np.mean(Py))
        avg_disp_px = float(0.5 * (np.mean(np.abs(dx_m)) + np.mean(np.abs(dy_m))))

    # Single scaling step to SI:
    s1 = float(k_force_per_m * pixel_size_m)            # N per pixel
    s2 = float(k_force_per_m * (pixel_size_m**2))       # N·m per pixel^2

    Fx = s1 * float(sum_dx)
    Fy = s1 * float(sum_dy)
    tau_z = s2 * float(tau_px2)

    return dict(
        Fx=Fx, Fy=Fy, tau_z=tau_z,
        cop_x=cop_x, cop_y=cop_y,
        n_used=int(dx_m.size),
        avg_disp_px=avg_disp_px,
    )
