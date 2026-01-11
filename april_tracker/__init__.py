"""AprilTag tracking package for comparing AprilTag pose with odometry."""
from .tracker_data_types import RigidBodyConfig, Pose, FrameResult
from .tracker import AprilTagTracker
from .alignment import compute_alignment_transform, compute_rotation_alignment
from .tracker_io import load_zed_frames, load_walk_log, find_closest_odom
from .processing import process_frames, compare_positions
from .visualization import plot_comparison, visualize_on_frames
from .config import create_tracker, TAG_SIZE_M, RIGID_BODIES, CAMERA_MATRIX, DIST_COEFFS

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
    'TAG_SIZE_M',
    'RIGID_BODIES',
    'CAMERA_MATRIX',
    'DIST_COEFFS',
]
