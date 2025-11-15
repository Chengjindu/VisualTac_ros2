"""
setting.py – configuration for GelWedge marker grid and tracking parameters
This module provides default calibration constants and initializes all global
variables used by the marker-tracking algorithm. It defines both the geometric
layout of the printed marker grid (x0, y0, dx, dy) and the numerical thresholds
used inside find_marker.so (e.g. theta, cost_ratio).

To apply new values, modify them below or pass a different rescale argument
when calling init(rescale=...).

All variables are declared as globals so they can be imported anywhere in the
ROS 2 package (e.g. gelwedge_demo_node).
"""

def init(rescale: float = 2.0):
    """
    Initialize all global constants used for GelWedge marker tracking.

    Args:
        rescale (float): optional image-to-real scaling factor used to adapt
                         pixel spacing for different camera resolutions.
                         Default = 2.0.
    """
    # Declare all globals to make them accessible module-wide
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    global theta, dmin_ratio, dmax_ratio, moving_max_ratio
    global flow_diff_ratio, cost_ratio, K1, K2, reference_dx

    # -------------------------------------------------------------------------
    # ----------------------------- GRID GEOMETRY -----------------------------
    # -------------------------------------------------------------------------
    RESCALE = rescale
    N_ = 10
    M_ = 10
    #   → Total grid size = N_ rows × M_ columns of printed markers

    fps_ = 30
    #   → Nominal camera frame rate used for timing the DFS search.
    #     Affects how long the matching algorithm is allowed to run per frame.

    # Grid origin (top-left corner of marker array) in image pixels.
    # These are base coordinates for the first marker, scaled by RESCALE.
    x0_ = 32 / RESCALE
    y0_ = 40 / RESCALE

    # Inter-marker spacing along horizontal and vertical directions (pixels).
    # These values define the expected distance between neighboring markers in
    # the static grid pattern. Larger spacing → wider grid and larger motion
    # threshold range.
    dx_ = 60 / RESCALE
    dy_ = 80 / RESCALE

    # -------------------------------------------------------------------------
    # -------------------------- MATCHING PARAMETERS --------------------------
    # -------------------------------------------------------------------------

    # Maximum angular deviation (degrees) between two markers that can still be
    # considered aligned along a grid row or column.  Smaller → stricter
    # alignment enforcement; larger → more tolerant to distortion or bending.
    theta = 45.0

    # Distance ratio bounds relative to dx used to validate neighbor spacing.
    #   dmin_ratio → lower bound multiplier
    #   dmax_ratio → upper bound multiplier
    # Distances between markers must satisfy:
    #     (dx * dmin_ratio)^2  ≤  distance^2  ≤  (dx * dmax_ratio)^2
    dmin_ratio = 0.7     # too small → may merge adjacent rows/cols
    dmax_ratio = 1.5     # too large → may match to wrong neighbors

    # Maximum displacement (in px) allowed for a marker between frames,
    # expressed as dx × moving_max_ratio.  Larger values tolerate faster
    # motion but may admit false matches.
    moving_max_ratio = 0.8 # reduce this helps to mitigate the choppy/freeze when the torque is getting big

    # Threshold (in px) for differences in local optical flow vectors between
    # adjacent markers.  If neighboring flow vectors differ by more than
    # dx × flow_diff_ratio, the configuration is penalized heavily.
    flow_diff_ratio = 0.8

    # Cost scaling factor controlling how strict the overall matching cost is.
    # A higher value raises the acceptable cost threshold (more tolerant);
    # lower values make the algorithm stricter but may fail under deformation.
    # The effective threshold is:
    #     cost_threshold = cost_ratio * (dx / reference_dx)^2
    cost_ratio = 15000.0

    # Weight coefficients in the matching cost function:
    #   K1 → penalizes absolute displacement from the original grid location.
    #   K2 → penalizes inconsistency of flow between neighboring markers.
    # Typically, K1 controls smoothness to the reference grid,
    # while K2 enforces local motion coherence.
    K1 = 0.1
    K2 = 1.0

    # Reference marker spacing (pixels) used to normalize cost_threshold.
    # For grids originally tuned with dx ≈ 21 px, keep reference_dx = 21.
    # If your printed marker spacing is larger, update accordingly so that the
    # cost scaling remains consistent across setups.
    reference_dx = 40.0

    # -------------------------------------------------------------------------
    # Print configuration summary for verification
    # -------------------------------------------------------------------------
    print(
        f"[GelWedge settings] Grid {N_}x{M_}, fps={fps_}, scale={RESCALE}\n"
        f"  dx={dx_:.2f}px  dy={dy_:.2f}px  theta={theta}°\n"
        f"  dmin_ratio={dmin_ratio}  dmax_ratio={dmax_ratio}\n"
        f"  moving_max_ratio={moving_max_ratio}  flow_diff_ratio={flow_diff_ratio}\n"
        f"  cost_ratio={cost_ratio}  K1={K1}  K2={K2}  reference_dx={reference_dx}"
    )
