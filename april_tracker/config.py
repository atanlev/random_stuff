"""Configuration for the AprilTag tracker."""
from __future__ import annotations

import numpy as np

from .tracker_data_types import RigidBodyConfig
from .tracker import AprilTagTracker

ZED_FRAMES_PATH = "zed_frames2.pkl"
WALK_LOG_PATH = "walk_log2.pkl"

# Tag configuration
TAG_SIZE_M = 0.116

RIGID_BODIES = [
    RigidBodyConfig(name="base_link", tag_id=75, tag_size=TAG_SIZE_M),
]

# Offset from AprilTag center to base_link origin (in robot body frame)
# AprilTag is 8cm forward (X) and 6cm up (Z) from base_link
# NOTE: This offset is used to transform the detected AprilTag position to base_link
# If the AprilTag is mounted FORWARD and UP from base_link, the offset should be POSITIVE
# AprilTag is 8cm forward (X) and 6cm up (Z) from base_link
# Set to zero - offset will be searched automatically
APRILTAG_TO_BASELINK_OFFSET = np.array([0.0, 0.0, 0.0])  # X=forward, Y=left, Z=up

# Default ZED camera intrinsics (used if not found in pickle file)
DEFAULT_CAMERA_MATRIX = np.array([
    [520.5241088867188, 0, 649.637939453125],
    [0, 520.5241088867188, 368.625732421875],
    [0, 0, 1]
], dtype=np.float64)
DEFAULT_DIST_COEFFS = np.zeros(5, dtype=np.float64)


def get_camera_params(intrinsics: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Get camera matrix and distortion coefficients from intrinsics dict or use defaults.

    Args:
        intrinsics: Dict with fx, fy, cx, cy, k1, k2, p1, p2, k3 or None

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: 5-element distortion coefficients array
    """
    if intrinsics is None:
        print("Using default camera intrinsics")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

    fx = intrinsics.get('fx', DEFAULT_CAMERA_MATRIX[0, 0])
    fy = intrinsics.get('fy', DEFAULT_CAMERA_MATRIX[1, 1])
    cx = intrinsics.get('cx', DEFAULT_CAMERA_MATRIX[0, 2])
    cy = intrinsics.get('cy', DEFAULT_CAMERA_MATRIX[1, 2])

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    k1 = intrinsics.get('k1', 0.0)
    k2 = intrinsics.get('k2', 0.0)
    p1 = intrinsics.get('p1', 0.0)
    p2 = intrinsics.get('p2', 0.0)
    k3 = intrinsics.get('k3', 0.0)

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    print(f"Using camera intrinsics from pickle: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    return camera_matrix, dist_coeffs

# Time offset configuration
# Set to False to disable automatic time offset detection
AUTO_TIME_OFFSET = False

# Odometry frame selection
# If None, assumes odometry entries have x, y, z, qx, qy, qz, qw directly
# If set to a string (e.g., "odom", "vision"), looks for odometry[i]["frame"] == ODOM_FRAME
ODOM_FRAME = 'base_link'

# Debug visualization options
DEBUG_AXIS_FLIP_X = False  # Flip X axis in visualization
DEBUG_AXIS_FLIP_Y = False  # Flip Y axis in visualization
DEBUG_AXIS_FLIP_Z = False  # Flip Z axis in visualization


def create_tracker(intrinsics: dict | None = None) -> AprilTagTracker:
    """Create an AprilTagTracker with configuration.

    Args:
        intrinsics: Camera intrinsics dict from pickle file, or None to use defaults
    """
    camera_matrix, dist_coeffs = get_camera_params(intrinsics)
    return AprilTagTracker(
        rigid_bodies=RIGID_BODIES,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

