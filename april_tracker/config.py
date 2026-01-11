"""Configuration for the AprilTag tracker."""
from __future__ import annotations

import numpy as np

from .data_types import RigidBodyConfig
from .tracker import AprilTagTracker


# Tag configuration
TAG_SIZE_M = 0.116

RIGID_BODIES = [
    RigidBodyConfig(name="base_link", tag_id=75, tag_size=TAG_SIZE_M),
]

# ZED camera intrinsics
CAMERA_MATRIX = np.array([
    [520.5241088867188, 0, 649.637939453125],
    [0, 520.5241088867188, 368.625732421875],
    [0, 0, 1]
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


def create_tracker() -> AprilTagTracker:
    """Create an AprilTagTracker with default configuration."""
    return AprilTagTracker(
        rigid_bodies=RIGID_BODIES,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
    )
