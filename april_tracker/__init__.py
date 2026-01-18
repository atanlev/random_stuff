"""AprilTag tracking package for comparing AprilTag pose with odometry."""
from .tracker_data_types import RigidBodyConfig, Pose, FrameResult
from .tracker import AprilTagTracker
from .alignment import compute_alignment_transform, compute_rotation_alignment, find_time_offset
from .tracker_io import load_zed_frames, load_walk_log, find_closest_odom
from .processing import process_frames, compare_positions
from .visualization import plot_comparison, visualize_on_frames
from .config import (
    create_tracker, get_camera_params, TAG_SIZE_M, RIGID_BODIES,
    DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS,
    APRILTAG_TO_BASELINK_OFFSET, AUTO_TIME_OFFSET, ZED_FRAMES_PATH, WALK_LOG_PATH
)

__all__ = [
    # Data types
    'RigidBodyConfig',
    'Pose',
    'FrameResult',
    # Tracker
    'AprilTagTracker',
    # Alignment
    'compute_alignment_transform',
    'compute_rotation_alignment',
    'find_time_offset',
    # I/O
    'load_zed_frames',
    'load_walk_log',
    'find_closest_odom',
    # Processing
    'process_frames',
    'compare_positions',
    # Visualization
    'plot_comparison',
    'visualize_on_frames',
    # Config
    'create_tracker',
    'get_camera_params',
    'TAG_SIZE_M',
    'RIGID_BODIES',
    'DEFAULT_CAMERA_MATRIX',
    'DEFAULT_DIST_COEFFS',
    'APRILTAG_TO_BASELINK_OFFSET',
    'AUTO_TIME_OFFSET',
    'ZED_FRAMES_PATH',
    'WALK_LOG_PATH',
]
