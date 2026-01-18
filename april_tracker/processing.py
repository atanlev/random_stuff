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
    Computes direct transformation from camera frame to odometry frame.
    Stops when frames go beyond odometry time range.

    Args:
        zed_frames: List of frame data with 'frame' and 'timestamp' keys
        walk_log: List of odometry entries with position and orientation
        tracker: AprilTagTracker instance
        time_offset_s: Optional time offset in seconds. If None, auto-detected.

    Returns:
        april_results: AprilTag tracking results in camera frame for each frame
        matched_odom: Corresponding odometry data (absolute positions) for each frame
        frames_used: The subset of zed_frames that were processed
        R: Rotation matrix to transform camera frame to odometry frame
        scale: Scale factor
        t: Translation vector (camera_to_odom: odom_pos = scale * R @ camera_pos + t)
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
                # Use raw AprilTag position for time offset detection
                april_positions_for_offset.append(base_link_pose.position)

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

    # Match odometry without computing relative positions
    # We'll compute the direct camera-to-odom transform instead
    matched_odom = []
    for frame_data in frames_used:
        # Apply time offset: if odom is delayed, we look at earlier odom
        corrected_timestamp = frame_data['timestamp'] - time_offset_s

        odom = find_closest_odom(corrected_timestamp, walk_log)
        if odom:
            odom_pos = np.array([odom['x'], odom['y'], odom['z']])
            odom_quat = np.array([odom['qx'], odom['qy'], odom['qz'], odom['qw']])
            matched_odom_entry = {
                'timestamp_utc': odom['timestamp_utc'],
                'position': odom_pos,  # Absolute position in odometry frame
                'quaternion': odom_quat,
                'time_diff': abs(odom['timestamp_utc'] - corrected_timestamp),
            }
            matched_odom.append(matched_odom_entry)
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
                    # Use RAW AprilTag position (WITHOUT offset) for alignment
                    # The offset will be applied AFTER alignment using the aligned rotation
                    stationary_april_pos.append(base_link_pose.position)
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

        # First align the RAW AprilTag position to world frame
        april_pos_raw = base_link_pose.position
        april_aligned_before_offset = scale * (R @ april_pos_raw) + t

        # Now apply the offset using the ALIGNED rotation
        # The offset should be rotated by the aligned orientation, not the raw one
        april_rot = Rotation.from_matrix(base_link_pose.rotation)
        april_rot_aligned = R_rot_align * april_rot
        offset_in_world = april_rot_aligned.as_matrix() @ APRILTAG_TO_BASELINK_OFFSET
        april_aligned = april_aligned_before_offset - offset_in_world

        # Convert AprilTag rotation matrix to quaternion
        april_rot = Rotation.from_matrix(base_link_pose.rotation)
        april_quat_raw = april_rot.as_quat()  # [x, y, z, w]
        # Apply rotation alignment
        april_rot_aligned = R_rot_align * april_rot
        april_quat_aligned = april_rot_aligned.as_quat()

        april_positions.append(april_pos_raw)  # Store raw position in camera frame
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
    april_eulers = []  # Store euler angles for correlation analysis
    odom_eulers = []
    for april_q, odom_q in zip(april_quats_aligned, odom_quats):
        # Compute relative rotation: R_diff = R_odom^-1 * R_april_aligned
        r_april = Rotation.from_quat(april_q)
        r_odom = Rotation.from_quat(odom_q)
        r_diff = r_odom.inv() * r_april
        # Get angle of rotation (magnitude of axis-angle)
        angle = np.abs(r_diff.magnitude()) * 180 / np.pi  # degrees
        angle_errors.append(angle)
        # Convert to euler angles (roll, pitch, yaw)
        april_eulers.append(r_april.as_euler('xyz', degrees=True))
        odom_eulers.append(r_odom.as_euler('xyz', degrees=True))
    angle_errors = np.array(angle_errors)
    april_eulers = np.array(april_eulers)
    odom_eulers = np.array(odom_eulers)

    # Filter outliers for statistics (based on position)
    inlier_mask = pos_error_norms < outlier_threshold
    n_outliers = np.sum(~inlier_mask)
    n_inliers = np.sum(inlier_mask)

    if n_inliers == 0:
        raise ValueError(
            f"All {len(pos_error_norms)} frames filtered as outliers (threshold={outlier_threshold}m). "
            f"Position errors range from {np.min(pos_error_norms):.3f}m to {np.max(pos_error_norms):.3f}m. "
            f"This suggests APRILTAG_TO_BASELINK_OFFSET may be incorrect. "
            f"Try adjusting the offset or increasing outlier_threshold."
        )

    pos_error_norms_filtered = pos_error_norms[inlier_mask]
    angle_errors_filtered = angle_errors[inlier_mask]

    # Filter position and euler data for correlation calculation
    april_positions_aligned_filtered = april_positions_aligned[inlier_mask]
    odom_positions_filtered = odom_positions[inlier_mask]
    april_eulers_filtered = april_eulers[inlier_mask]
    odom_eulers_filtered = odom_eulers[inlier_mask]

    # Compute correlation scores for position (X, Y, Z)
    corr_x = np.corrcoef(april_positions_aligned_filtered[:, 0], odom_positions_filtered[:, 0])[0, 1]
    corr_y = np.corrcoef(april_positions_aligned_filtered[:, 1], odom_positions_filtered[:, 1])[0, 1]
    corr_z = np.corrcoef(april_positions_aligned_filtered[:, 2], odom_positions_filtered[:, 2])[0, 1]

    # Compute correlation scores for orientation using quaternions
    # Quaternions are better than Euler angles (no gimbal lock, no wrapping issues)
    april_quats_aligned_filtered = april_quats_aligned[inlier_mask]
    odom_quats_filtered = odom_quats[inlier_mask]

    # Compute frame-to-frame angular velocities (rotation differences)
    # This measures how well the tracker follows rotational changes
    if len(april_quats_aligned_filtered) > 10:
        april_angular_vel = []
        odom_angular_vel = []

        for i in range(1, len(april_quats_aligned_filtered)):
            # Compute relative rotation between consecutive frames
            r_april_prev = Rotation.from_quat(april_quats_aligned_filtered[i-1])
            r_april_curr = Rotation.from_quat(april_quats_aligned_filtered[i])
            r_odom_prev = Rotation.from_quat(odom_quats_filtered[i-1])
            r_odom_curr = Rotation.from_quat(odom_quats_filtered[i])

            # Angular change between frames
            april_delta = r_april_curr * r_april_prev.inv()
            odom_delta = r_odom_curr * r_odom_prev.inv()

            # Convert to axis-angle to get angular velocity components
            april_rotvec = april_delta.as_rotvec()  # [rad]
            odom_rotvec = odom_delta.as_rotvec()

            april_angular_vel.append(april_rotvec)
            odom_angular_vel.append(odom_rotvec)

        april_angular_vel = np.array(april_angular_vel)
        odom_angular_vel = np.array(odom_angular_vel)

        # Compute correlation for each axis of angular velocity
        try:
            if np.std(april_angular_vel[:, 0]) > 1e-5 and np.std(odom_angular_vel[:, 0]) > 1e-5:
                corr_wx = np.corrcoef(april_angular_vel[:, 0], odom_angular_vel[:, 0])[0, 1]
            else:
                corr_wx = np.nan
        except (ValueError, RuntimeWarning):
            corr_wx = np.nan

        try:
            if np.std(april_angular_vel[:, 1]) > 1e-5 and np.std(odom_angular_vel[:, 1]) > 1e-5:
                corr_wy = np.corrcoef(april_angular_vel[:, 1], odom_angular_vel[:, 1])[0, 1]
            else:
                corr_wy = np.nan
        except (ValueError, RuntimeWarning):
            corr_wy = np.nan

        try:
            if np.std(april_angular_vel[:, 2]) > 1e-5 and np.std(odom_angular_vel[:, 2]) > 1e-5:
                corr_wz = np.corrcoef(april_angular_vel[:, 2], odom_angular_vel[:, 2])[0, 1]
            else:
                corr_wz = np.nan
        except (ValueError, RuntimeWarning):
            corr_wz = np.nan

        # Also compute correlation on absolute quaternion components for reference
        try:
            corr_qx = np.corrcoef(april_quats_aligned_filtered[:, 0], odom_quats_filtered[:, 0])[0, 1]
        except (ValueError, RuntimeWarning):
            corr_qx = np.nan
        try:
            corr_qy = np.corrcoef(april_quats_aligned_filtered[:, 1], odom_quats_filtered[:, 1])[0, 1]
        except (ValueError, RuntimeWarning):
            corr_qy = np.nan
        try:
            corr_qz = np.corrcoef(april_quats_aligned_filtered[:, 2], odom_quats_filtered[:, 2])[0, 1]
        except (ValueError, RuntimeWarning):
            corr_qz = np.nan
        try:
            corr_qw = np.corrcoef(april_quats_aligned_filtered[:, 3], odom_quats_filtered[:, 3])[0, 1]
        except (ValueError, RuntimeWarning):
            corr_qw = np.nan
    else:
        corr_wx = np.nan
        corr_wy = np.nan
        corr_wz = np.nan
        corr_qx = np.nan
        corr_qy = np.nan
        corr_qz = np.nan
        corr_qw = np.nan

    # Also compute Euler angles for human-readable variance reporting
    roll_std_april = np.std(april_eulers_filtered[:, 0])
    roll_std_odom = np.std(odom_eulers_filtered[:, 0])
    pitch_std_april = np.std(april_eulers_filtered[:, 1])
    pitch_std_odom = np.std(odom_eulers_filtered[:, 1])
    yaw_std_april = np.std(april_eulers_filtered[:, 2])
    yaw_std_odom = np.std(odom_eulers_filtered[:, 2])

    # Map angular velocity correlations to Roll/Pitch/Yaw
    # This is more meaningful than absolute orientation correlation
    corr_roll = corr_wx
    corr_pitch = corr_wy
    corr_yaw = corr_wz

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
    print("\nCorrelation scores:")
    print(f"  X: {corr_x:.4f}")
    print(f"  Y: {corr_y:.4f}")
    print(f"  Z: {corr_z:.4f}")

    print("\n=== Orientation Comparison ===")
    print(f"Mean error:        {mean_angle_err:.1f} deg")
    print(f"Max error:         {max_angle_err:.1f} deg")
    print(f"Std:               {std_angle_err:.1f} deg")
    print(f"50th percentile:   {p50_angle:.1f} deg")
    print(f"90th percentile:   {p90_angle:.1f} deg")
    print(f"99th percentile:   {p99_angle:.1f} deg")
    print("\nOrientation variance (std dev in degrees):")
    print(f"  Roll:  AprilTag={roll_std_april:.2f}, Odom={roll_std_odom:.2f}")
    print(f"  Pitch: AprilTag={pitch_std_april:.2f}, Odom={pitch_std_odom:.2f}")
    print(f"  Yaw:   AprilTag={yaw_std_april:.2f}, Odom={yaw_std_odom:.2f}")

    # Show range analysis for Euler angles
    print("\nEuler angle ranges (degrees):")
    april_roll_min, april_roll_max = np.min(april_eulers_filtered[:, 0]), np.max(april_eulers_filtered[:, 0])
    april_pitch_min, april_pitch_max = np.min(april_eulers_filtered[:, 1]), np.max(april_eulers_filtered[:, 1])
    april_yaw_min, april_yaw_max = np.min(april_eulers_filtered[:, 2]), np.max(april_eulers_filtered[:, 2])
    odom_roll_min, odom_roll_max = np.min(odom_eulers_filtered[:, 0]), np.max(odom_eulers_filtered[:, 0])
    odom_pitch_min, odom_pitch_max = np.min(odom_eulers_filtered[:, 1]), np.max(odom_eulers_filtered[:, 1])
    odom_yaw_min, odom_yaw_max = np.min(odom_eulers_filtered[:, 2]), np.max(odom_eulers_filtered[:, 2])

    april_roll_mean = np.mean(april_eulers_filtered[:, 0])
    april_pitch_mean = np.mean(april_eulers_filtered[:, 1])
    april_yaw_mean = np.mean(april_eulers_filtered[:, 2])
    odom_roll_mean = np.mean(odom_eulers_filtered[:, 0])
    odom_pitch_mean = np.mean(odom_eulers_filtered[:, 1])
    odom_yaw_mean = np.mean(odom_eulers_filtered[:, 2])

    print(f"  Roll:  April [{april_roll_min:6.2f}, {april_roll_max:6.2f}] mean={april_roll_mean:6.2f}  "
          f"Odom [{odom_roll_min:6.2f}, {odom_roll_max:6.2f}] mean={odom_roll_mean:6.2f}")
    print(f"  Pitch: April [{april_pitch_min:6.2f}, {april_pitch_max:6.2f}] mean={april_pitch_mean:6.2f}  "
          f"Odom [{odom_pitch_min:6.2f}, {odom_pitch_max:6.2f}] mean={odom_pitch_mean:6.2f}")
    print(f"  Yaw:   April [{april_yaw_min:6.2f}, {april_yaw_max:6.2f}] mean={april_yaw_mean:6.2f}  "
          f"Odom [{odom_yaw_min:6.2f}, {odom_yaw_max:6.2f}] mean={odom_yaw_mean:6.2f}")

    # Show angular velocity statistics
    if len(april_quats_aligned_filtered) > 10:
        print("\nAngular velocity statistics (rad/frame):")
        print(f"  wx (roll):  April std={np.std(april_angular_vel[:, 0]):.6f}, "
              f"Odom std={np.std(odom_angular_vel[:, 0]):.6f}")
        print(f"  wy (pitch): April std={np.std(april_angular_vel[:, 1]):.6f}, "
              f"Odom std={np.std(odom_angular_vel[:, 1]):.6f}")
        print(f"  wz (yaw):   April std={np.std(april_angular_vel[:, 2]):.6f}, "
              f"Odom std={np.std(odom_angular_vel[:, 2]):.6f}")

        print("\nAngular velocity ranges (rad/frame):")
        april_wx_min, april_wx_max = np.min(april_angular_vel[:, 0]), np.max(april_angular_vel[:, 0])
        april_wy_min, april_wy_max = np.min(april_angular_vel[:, 1]), np.max(april_angular_vel[:, 1])
        april_wz_min, april_wz_max = np.min(april_angular_vel[:, 2]), np.max(april_angular_vel[:, 2])
        odom_wx_min, odom_wx_max = np.min(odom_angular_vel[:, 0]), np.max(odom_angular_vel[:, 0])
        odom_wy_min, odom_wy_max = np.min(odom_angular_vel[:, 1]), np.max(odom_angular_vel[:, 1])
        odom_wz_min, odom_wz_max = np.min(odom_angular_vel[:, 2]), np.max(odom_angular_vel[:, 2])

        april_wx_mean = np.mean(april_angular_vel[:, 0])
        april_wy_mean = np.mean(april_angular_vel[:, 1])
        april_wz_mean = np.mean(april_angular_vel[:, 2])
        odom_wx_mean = np.mean(odom_angular_vel[:, 0])
        odom_wy_mean = np.mean(odom_angular_vel[:, 1])
        odom_wz_mean = np.mean(odom_angular_vel[:, 2])

        print(f"  wx (roll):  April [{april_wx_min:8.5f}, {april_wx_max:8.5f}] mean={april_wx_mean:8.5f}  "
              f"Odom [{odom_wx_min:8.5f}, {odom_wx_max:8.5f}] mean={odom_wx_mean:8.5f}")
        print(f"  wy (pitch): April [{april_wy_min:8.5f}, {april_wy_max:8.5f}] mean={april_wy_mean:8.5f}  "
              f"Odom [{odom_wy_min:8.5f}, {odom_wy_max:8.5f}] mean={odom_wy_mean:8.5f}")
        print(f"  wz (yaw):   April [{april_wz_min:8.5f}, {april_wz_max:8.5f}] mean={april_wz_mean:8.5f}  "
              f"Odom [{odom_wz_min:8.5f}, {odom_wz_max:8.5f}] mean={odom_wz_mean:8.5f}")
    print("\nAngular velocity correlation (frame-to-frame rotation changes):")
    if np.isnan(corr_roll):
        print("  Roll (wx):  N/A")
    else:
        print(f"  Roll (wx):  {corr_roll:.4f}")
    if np.isnan(corr_pitch):
        print("  Pitch (wy): N/A")
    else:
        print(f"  Pitch (wy): {corr_pitch:.4f}")
    if np.isnan(corr_yaw):
        print("  Yaw (wz):   N/A")
    else:
        print(f"  Yaw (wz):   {corr_yaw:.4f}")
    print("\nAbsolute quaternion correlation (reference only):")
    print(f"  qx: {corr_qx:.4f}, qy: {corr_qy:.4f}, qz: {corr_qz:.4f}, qw: {corr_qw:.4f}")

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
        'correlation_x': corr_x,
        'correlation_y': corr_y,
        'correlation_z': corr_z,
        'correlation_qx': corr_qx,
        'correlation_qy': corr_qy,
        'correlation_qz': corr_qz,
        'correlation_qw': corr_qw,
        'correlation_roll': corr_roll,  # Angular velocity correlation (wx)
        'correlation_pitch': corr_pitch,  # Angular velocity correlation (wy)
        'correlation_yaw': corr_yaw,  # Angular velocity correlation (wz)
    }
