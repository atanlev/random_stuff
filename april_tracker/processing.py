"""Frame processing and comparison functions."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .tracker_data_types import FrameResult
from .tracker import AprilTagTracker
from .tracker_io import find_closest_odom
from .alignment import compute_alignment_transform, compute_rotation_alignment, find_time_offset
from .config import APRILTAG_TO_BASELINK_OFFSET


def apply_tag_to_baselink_offset(april_pos: np.ndarray, april_rot: np.ndarray) -> np.ndarray:
    """
    Transform AprilTag position to base_link position using the known offset.

    The AprilTag is mounted at an offset from the base_link origin.
    To get base_link position: base_link_pos = april_pos - R_april @ offset

    Args:
        april_pos: AprilTag position in reference frame
        april_rot: AprilTag rotation matrix (3x3)

    Returns:
        base_link position in reference frame
    """
    # Offset is defined in body frame, rotate it to reference frame
    offset_in_ref = april_rot @ APRILTAG_TO_BASELINK_OFFSET
    # Base_link is at april_pos minus the rotated offset
    return april_pos - offset_in_ref


def process_frames(
    zed_frames: list[dict],
    walk_log: list[dict],
    tracker: AprilTagTracker,
    time_offset_s: float | None = None,
) -> tuple[list[FrameResult], list[dict], list[dict], np.ndarray, float, np.ndarray, Rotation, float]:
    """
    Process frames from zed_frames.pkl that overlap with walk_log odometry.
    Stops when frames go beyond odometry time range.

    Args:
        zed_frames: List of frame data with 'frame' and 'timestamp' keys
        walk_log: List of odometry entries with position and orientation
        tracker: AprilTagTracker instance
        time_offset_s: Optional time offset in seconds. If None, auto-detected.

    Returns:
        april_results: AprilTag tracking results for each frame
        matched_odom: Corresponding odometry data for each frame
        frames_used: The subset of zed_frames that were processed
        R: Rotation matrix to align AprilTag positions to odometry frame
        scale: Scale factor
        t: Translation vector
        R_rot_align: Rotation to align AprilTag orientations to odometry
        time_offset_s: The time offset used (auto-detected or provided)
    """
    # Find first frame where walk_log has data (timestamps overlap)
    walk_log_start = walk_log[0]['timestamp_utc']
    walk_log_end = walk_log[-1]['timestamp_utc']

    # Find first frame within walk_log time range for reference
    ref_frame_idx = None
    for i, frame_data in enumerate(zed_frames):
        if walk_log_start <= frame_data['timestamp'] <= walk_log_end:
            ref_frame_idx = i
            break

    if ref_frame_idx is None:
        raise ValueError("No frames overlap with walk_log timestamps")

    print(f"Using frame {ref_frame_idx} as reference (first frame in walk_log range)")

    # Set reference from first valid frame
    if not tracker.set_reference_from_frame(zed_frames[ref_frame_idx]['frame']):
        raise ValueError("Failed to set reference - base_link tag not visible")

    # First pass: process all frames to get AprilTag positions
    print("First pass: detecting AprilTags...")
    april_results = []
    frames_used = []

    for frame_idx, frame_data in enumerate(zed_frames):
        timestamp = frame_data['timestamp']

        # Stop when frames go beyond odometry time range
        if timestamp > walk_log_end:
            print(f"  Stopping at frame {frame_idx} (beyond odometry time range)")
            break

        frame = frame_data['frame']
        result = tracker.process_frame(frame, frame_idx, timestamp)
        april_results.append(result)
        frames_used.append(frame_data)

        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx}/{len(zed_frames)}")

    print(f"Processed {len(april_results)} frames")

    # Find time offset if not provided
    if time_offset_s is None:
        print("\nFinding optimal time offset...")
        # Collect AprilTag positions and timestamps for offset detection
        april_timestamps_for_offset = []
        april_positions_for_offset = []

        for result, frame_data in zip(april_results, frames_used):
            base_link_pose = result.poses.get('base_link')
            if base_link_pose is not None:
                april_timestamps_for_offset.append(frame_data['timestamp'])
                baselink_pos = apply_tag_to_baselink_offset(
                    base_link_pose.position, base_link_pose.rotation
                )
                april_positions_for_offset.append(baselink_pos)

        if len(april_timestamps_for_offset) >= 20:
            april_timestamps_arr = np.array(april_timestamps_for_offset)
            april_positions_arr = np.array(april_positions_for_offset)

            # Get raw odometry positions and timestamps
            odom_timestamps = np.array([o['timestamp_utc'] for o in walk_log])
            odom_positions = np.array([[o['x'], o['y'], o['z']] for o in walk_log])

            time_offset_s, offset_error = find_time_offset(
                april_timestamps_arr, april_positions_arr,
                odom_timestamps, odom_positions,
                search_range_ms=100.0, step_ms=2.0
            )
            print(f"  Detected time offset: {time_offset_s * 1000:.1f} ms (error: {offset_error:.4f}m)")
        else:
            print("  Not enough AprilTag detections for offset detection, using 0ms")
            time_offset_s = 0.0
    else:
        print(f"Using provided time offset: {time_offset_s * 1000:.1f} ms")

    # Second pass: match odometry with corrected timestamps
    print(f"\nMatching odometry with time offset correction...")

    # Get reference odometry at corrected timestamp
    ref_timestamp = zed_frames[ref_frame_idx]['timestamp'] - time_offset_s
    ref_odom = find_closest_odom(ref_timestamp, walk_log)
    if ref_odom is None:
        # Fall back to finding reference without offset correction
        ref_timestamp = zed_frames[ref_frame_idx]['timestamp']
        ref_odom = find_closest_odom(ref_timestamp, walk_log)
        print(f"  Warning: corrected ref timestamp outside odom range, using uncorrected")
    ref_odom_pos = np.array([ref_odom['x'], ref_odom['y'], ref_odom['z']])

    print(f"Reference odometry position: {ref_odom_pos}")

    matched_odom = []
    for frame_data in frames_used:
        # Apply time offset: if odom is delayed, we look at earlier odom
        corrected_timestamp = frame_data['timestamp'] - time_offset_s

        odom = find_closest_odom(corrected_timestamp, walk_log)
        if odom:
            odom_pos = np.array([odom['x'], odom['y'], odom['z']])
            odom_quat = np.array([odom['qx'], odom['qy'], odom['qz'], odom['qw']])
            relative_odom = {
                'timestamp_utc': odom['timestamp_utc'],
                'position': odom_pos - ref_odom_pos,
                'quaternion': odom_quat,
                'time_diff': abs(odom['timestamp_utc'] - corrected_timestamp),
            }
            matched_odom.append(relative_odom)
        else:
            matched_odom.append(None)

    print(f"Processed {len(april_results)} frames")

    # Detect stationary frames using odometry velocity
    stationary_threshold = 0.02  # m/s - below this is considered stationary

    stationary_april_pos = []
    stationary_odom_pos = []
    stationary_april_quats = []
    stationary_odom_quats = []

    prev_odom_pos = None
    prev_odom_time = None

    for result, odom in zip(april_results, matched_odom):
        if odom is None:
            prev_odom_pos = None
            prev_odom_time = None
            continue

        base_link_pose = result.poses.get('base_link')
        if base_link_pose is None:
            prev_odom_pos = None
            prev_odom_time = None
            continue

        odom_pos = odom['position']
        odom_time = odom['timestamp_utc']

        if prev_odom_pos is not None and prev_odom_time is not None:
            dt = odom_time - prev_odom_time
            if dt > 0:
                velocity = np.linalg.norm(odom_pos[:2] - prev_odom_pos[:2]) / dt
                if velocity < stationary_threshold:
                    # Apply offset to get base_link position from AprilTag position
                    baselink_pos = apply_tag_to_baselink_offset(
                        base_link_pose.position, base_link_pose.rotation
                    )
                    stationary_april_pos.append(baselink_pos)
                    stationary_odom_pos.append(odom_pos)
                    # Also collect quaternions for rotation alignment
                    april_quat = Rotation.from_matrix(base_link_pose.rotation).as_quat()
                    stationary_april_quats.append(april_quat)
                    stationary_odom_quats.append(odom['quaternion'])

        prev_odom_pos = odom_pos
        prev_odom_time = odom_time

    print(f"Found {len(stationary_april_pos)} stationary frames for alignment")

    if len(stationary_april_pos) < 10:
        print("Warning: Not enough stationary frames for alignment, using identity transform")
        R = np.eye(3)
        scale = 1.0
        t = np.zeros(3)
        R_rot_align = Rotation.identity()
    else:
        stationary_april_pos = np.array(stationary_april_pos)
        stationary_odom_pos = np.array(stationary_odom_pos)

        R, scale, t, inlier_mask = compute_alignment_transform(stationary_april_pos, stationary_odom_pos)
        n_inliers = np.sum(inlier_mask)
        n_outliers = len(inlier_mask) - n_inliers
        print("Position alignment transform computed from stationary frames:")
        print(f"  Inliers: {n_inliers}, Outliers removed: {n_outliers}")
        print(f"  Scale: {scale:.4f}")
        print(f"  Translation: {t}")

        # Compute rotation alignment using inlier stationary frames
        inlier_april_quats = [q for q, m in zip(stationary_april_quats, inlier_mask) if m]
        inlier_odom_quats = [q for q, m in zip(stationary_odom_quats, inlier_mask) if m]
        R_rot_align = compute_rotation_alignment(inlier_april_quats, inlier_odom_quats)
        print(f"Rotation alignment: {R_rot_align.as_euler('xyz', degrees=True)} (euler xyz deg)")

    return april_results, matched_odom, frames_used, R, scale, t, R_rot_align, time_offset_s


def compare_positions(
    april_results: list[FrameResult],
    matched_odom: list[dict],
    R: np.ndarray,
    scale: float,
    t: np.ndarray,
    R_rot_align: Rotation,
    outlier_threshold: float = 0.5,
) -> dict:
    """
    Compare AprilTag base_link positions and orientations with odometry.
    Applies alignment transform (rotation, scale, translation) to AprilTag positions.
    Applies rotation alignment to AprilTag orientations.
    Filters outliers based on error threshold.

    Returns dict with comparison data.
    """
    april_positions = []
    april_positions_aligned = []
    odom_positions = []
    april_quats = []
    april_quats_aligned = []
    odom_quats = []
    timestamps = []

    for result, odom in zip(april_results, matched_odom):
        if odom is None:
            continue

        base_link_pose = result.poses.get('base_link')
        if base_link_pose is None:
            continue

        # Apply offset to get base_link position from AprilTag position
        april_pos_raw = base_link_pose.position
        april_pos = apply_tag_to_baselink_offset(april_pos_raw, base_link_pose.rotation)
        april_aligned = scale * (R @ april_pos) + t

        # Convert AprilTag rotation matrix to quaternion
        april_rot = Rotation.from_matrix(base_link_pose.rotation)
        april_quat_raw = april_rot.as_quat()  # [x, y, z, w]
        # Apply rotation alignment
        april_rot_aligned = R_rot_align * april_rot
        april_quat_aligned = april_rot_aligned.as_quat()

        april_positions.append(april_pos)
        april_positions_aligned.append(april_aligned)
        odom_positions.append(odom['position'])
        april_quats.append(april_quat_raw)
        april_quats_aligned.append(april_quat_aligned)
        odom_quats.append(odom['quaternion'])
        timestamps.append(result.timestamp_utc)

    april_positions = np.array(april_positions)
    april_positions_aligned = np.array(april_positions_aligned)
    odom_positions = np.array(odom_positions)
    april_quats = np.array(april_quats)
    april_quats_aligned = np.array(april_quats_aligned)
    odom_quats = np.array(odom_quats)
    timestamps = np.array(timestamps)

    # Compute position errors (XYZ)
    pos_errors = april_positions_aligned - odom_positions
    pos_error_norms = np.linalg.norm(pos_errors, axis=1)

    # Compute orientation errors (angular difference in degrees) using aligned quaternions
    angle_errors = []
    for april_q, odom_q in zip(april_quats_aligned, odom_quats):
        # Compute relative rotation: R_diff = R_odom^-1 * R_april_aligned
        r_april = Rotation.from_quat(april_q)
        r_odom = Rotation.from_quat(odom_q)
        r_diff = r_odom.inv() * r_april
        # Get angle of rotation (magnitude of axis-angle)
        angle = np.abs(r_diff.magnitude()) * 180 / np.pi  # degrees
        angle_errors.append(angle)
    angle_errors = np.array(angle_errors)

    # Filter outliers for statistics (based on position)
    inlier_mask = pos_error_norms < outlier_threshold
    n_outliers = np.sum(~inlier_mask)

    pos_error_norms_filtered = pos_error_norms[inlier_mask]
    angle_errors_filtered = angle_errors[inlier_mask]

    # Compute position statistics
    mean_pos_err = np.mean(pos_error_norms_filtered) * 100  # cm
    max_pos_err = np.max(pos_error_norms_filtered) * 100
    std_pos_err = np.std(pos_error_norms_filtered) * 100
    p50_pos = np.percentile(pos_error_norms_filtered, 50) * 100
    p90_pos = np.percentile(pos_error_norms_filtered, 90) * 100
    p99_pos = np.percentile(pos_error_norms_filtered, 99) * 100

    # Compute orientation statistics
    mean_angle_err = np.mean(angle_errors_filtered)
    max_angle_err = np.max(angle_errors_filtered)
    std_angle_err = np.std(angle_errors_filtered)
    p50_angle = np.percentile(angle_errors_filtered, 50)
    p90_angle = np.percentile(angle_errors_filtered, 90)
    p99_angle = np.percentile(angle_errors_filtered, 99)

    print("\n=== Position Comparison (after alignment) ===")
    print(f"Total frames: {len(april_positions)}, Outliers removed: {n_outliers}")
    print(f"Mean error:        {mean_pos_err:.1f} cm")
    print(f"Max error:         {max_pos_err:.1f} cm")
    print(f"Std:               {std_pos_err:.1f} cm")
    print(f"50th percentile:   {p50_pos:.1f} cm")
    print(f"90th percentile:   {p90_pos:.1f} cm")
    print(f"99th percentile:   {p99_pos:.1f} cm")

    print("\n=== Orientation Comparison ===")
    print(f"Mean error:        {mean_angle_err:.1f} deg")
    print(f"Max error:         {max_angle_err:.1f} deg")
    print(f"Std:               {std_angle_err:.1f} deg")
    print(f"50th percentile:   {p50_angle:.1f} deg")
    print(f"90th percentile:   {p90_angle:.1f} deg")
    print(f"99th percentile:   {p99_angle:.1f} deg")

    return {
        'april_positions': april_positions_aligned[inlier_mask],
        'april_positions_raw': april_positions[inlier_mask],
        'odom_positions': odom_positions[inlier_mask],
        'april_quats': april_quats[inlier_mask],
        'april_quats_aligned': april_quats_aligned[inlier_mask],
        'odom_quats': odom_quats[inlier_mask],
        'timestamps': timestamps[inlier_mask],
        'pos_errors': pos_errors[inlier_mask],
        'pos_error_norms': pos_error_norms_filtered,
        'angle_errors': angle_errors_filtered,
        'inlier_mask': inlier_mask,
    }
