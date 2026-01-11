"""Configuration for the AprilTag tracker."""
from __future__ import annotations

import numpy as np

from .tracker_data_types import RigidBodyConfig
from .tracker import AprilTagTracker

ZED_FRAMES_PATH = "zed_frames.pkl"
WALK_LOG_PATH = "walk_log.pkl"

# Tag configuration
TAG_SIZE_M = 0.116

RIGID_BODIES = [
    RigidBodyConfig(name="base_link", tag_id=75, tag_size=TAG_SIZE_M),
]

# Offset from AprilTag center to base_link origin (in robot body frame)
# AprilTag is 8cm forward (X) and 6cm up (Z) from base_link
APRILTAG_TO_BASELINK_OFFSET = np.array([0.08, 0.01, 0.06])

# ZED camera intrinsics
CAMERA_MATRIX = np.array([
    [520.5241088867188, 0, 649.637939453125],
    [0, 520.5241088867188, 368.625732421875],
    [0, 0, 1]
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)

# Time offset configuration
# Set to False to disable automatic time offset detection
AUTO_TIME_OFFSET = True


def create_tracker() -> AprilTagTracker:
    """Create an AprilTagTracker with default configuration."""
    return AprilTagTracker(
        rigid_bodies=RIGID_BODIES,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
    )

