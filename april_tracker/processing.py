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
) -> tuple[list[FrameResult], list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Process frames from zed_frames.pkl that overlap with walk_log odometry.
    Computes rigid transformation from camera frame to odometry frame.
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
        t: Translation vector (camera_to_odom: odom_pos = R @ camera_pos + t)
        R_rot_align: 3x3 rotation matrix to align AprilTag orientations to odometry
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
    stationary_threshold = 0.2  # m/s - below this is considered stationary

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

    # Analyze raw axis correlations to find correct mapping
    if len(stationary_april_quats) >= 10:
        print("\n=== RAW AXIS CORRELATION ANALYSIS ===")
        print("This shows correlation between each AprilTag axis and each Odom axis")
        print("Look for high absolute correlations (close to +1 or -1) to find axis mapping\n")

        # Convert to euler angles
        april_eulers_raw = np.array([Rotation.from_quat(q).as_euler('xyz', degrees=True)
                                      for q in stationary_april_quats])
        odom_eulers_raw = np.array([Rotation.from_quat(q).as_euler('xyz', degrees=True)
                                     for q in stationary_odom_quats])

        axis_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']

        # Print correlation matrix
        print("Correlation Matrix (AprilTag vs Odom):")
        print(f"{'':12} | {'Odom Roll':>10} {'Odom Pitch':>11} {'Odom Yaw':>10}")
        print("-" * 50)
        for i, april_axis in enumerate(axis_names):
            row = f"{april_axis:12} |"
            for j in range(3):
                corr = np.corrcoef(april_eulers_raw[:, i], odom_eulers_raw[:, j])[0, 1]
                row += f" {corr:>10.3f}"
            print(row)

        print("\nValue ranges:")
        for i, name in enumerate(axis_names):
            april_min, april_max = np.min(april_eulers_raw[:, i]), np.max(april_eulers_raw[:, i])
            odom_min, odom_max = np.min(odom_eulers_raw[:, i]), np.max(odom_eulers_raw[:, i])
            april_std = np.std(april_eulers_raw[:, i])
            odom_std = np.std(odom_eulers_raw[:, i])
            print(f"  {name}: April [{april_min:7.2f}, {april_max:7.2f}] std={april_std:5.2f}  "
                  f"Odom [{odom_min:7.2f}, {odom_max:7.2f}] std={odom_std:5.2f}")
        print("=" * 50 + "\n")

    if len(stationary_april_pos) < 10:
        print("Warning: Not enough stationary frames for alignment, using identity transform")
        R = np.eye(3)
        t = np.zeros(3)
        R_rot_align = None  # No rotation alignment
    else:
        stationary_april_pos = np.array(stationary_april_pos)
        stationary_odom_pos = np.array(stationary_odom_pos)

        R, t, inlier_mask = compute_alignment_transform(stationary_april_pos, stationary_odom_pos)
        n_inliers = np.sum(inlier_mask)
        n_outliers = len(inlier_mask) - n_inliers
        print("Position alignment transform computed from stationary frames:")
        print(f"  Inliers: {n_inliers}, Outliers removed: {n_outliers}")
        print(f"  Translation: {t}")

        # Compute rotation alignment using inlier stationary frames
        inlier_april_quats = [q for q, m in zip(stationary_april_quats, inlier_mask) if m]
        inlier_odom_quats = [q for q, m in zip(stationary_odom_quats, inlier_mask) if m]
        euler_map = compute_rotation_alignment(inlier_april_quats, inlier_odom_quats)
        R_rot_align = euler_map  # Just the euler map now

    return april_results, matched_odom, frames_used, R, t, R_rot_align, time_offset_s


def find_best_offset(
    april_results: list,
    matched_odom: list,
    R: np.ndarray,
    t: np.ndarray,
    R_rot_align: np.ndarray,
    search_range: float = 0.15,  # Search +/- 15cm
    step: float = 0.02,  # 2cm steps
) -> np.ndarray:
    """
    Search for the best tag-to-baselink offset by minimizing position error.

    Returns:
        Best offset as numpy array [x, y, z]
    """
    best_offset = np.array([0.0, 0.0, 0.0])
    best_error = float('inf')

    # Search over X and Z (Y is typically 0 for centered tags)
    for ox in np.arange(-search_range, search_range + step, step):
        for oz in np.arange(-search_range, search_range + step, step):
            offset = np.array([ox, 0.0, oz])

            # Compute position error with this offset
            errors = []
            for result, odom in zip(april_results, matched_odom):
                if odom is None:
                    continue
                base_link_pose = result.poses.get('base_link')
                if base_link_pose is None:
                    continue

                april_pos_raw = base_link_pose.position
                april_aligned_before_offset = R @ april_pos_raw + t

                april_rot = Rotation.from_matrix(base_link_pose.rotation)
                april_rot_aligned = apply_rotation_alignment(april_rot, R_rot_align)
                offset_in_world = april_rot_aligned.as_matrix() @ offset
                april_aligned = april_aligned_before_offset - offset_in_world

                odom_pos = odom['position']
                error = np.linalg.norm(april_aligned - odom_pos)
                if error < 0.5:  # Only count inliers
                    errors.append(error)

            if len(errors) > 10:
                mean_error = np.mean(errors)
                if mean_error < best_error:
                    best_error = mean_error
                    best_offset = offset.copy()

    print(f"\n  Best offset found: [{best_offset[0]:.3f}, {best_offset[1]:.3f}, {best_offset[2]:.3f}] m")
    print(f"  Position error with offset: {best_error * 100:.1f} cm")

    return best_offset


def apply_rotation_alignment(april_rot: Rotation, R_align: np.ndarray | None) -> Rotation:
    """
    Apply rotation alignment to an AprilTag rotation.

    Args:
        april_rot: Raw AprilTag rotation
        R_align: 3x3 rotation matrix for alignment, or None for identity

    Returns:
        Aligned rotation: R_align @ R_april
    """
    if R_align is None:
        return april_rot

    # Apply rotation alignment: R_odom = R_align @ R_april
    return Rotation.from_matrix(R_align) * april_rot


def compare_positions(
    april_results: list[FrameResult],
    matched_odom: list[dict],
    R: np.ndarray,
    t: np.ndarray,
    R_rot_align: np.ndarray,
    outlier_threshold: float = 0.5,
    tag_offset: np.ndarray = None,
) -> dict:
    """
    Compare AprilTag base_link positions and orientations with odometry.
    Applies rigid alignment transform (rotation, translation) to AprilTag positions.
    Applies rotation alignment to AprilTag orientations.
    Filters outliers based on error threshold.

    Args:
        tag_offset: Optional offset from tag to base_link. If None, uses config value.

    Returns dict with comparison data.
    """
    if tag_offset is None:
        tag_offset = APRILTAG_TO_BASELINK_OFFSET

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

        # First align the RAW AprilTag position to world frame (rigid: no scale)
        april_pos_raw = base_link_pose.position
        april_aligned_before_offset = R @ april_pos_raw + t

        # Now apply the offset using the ALIGNED rotation
        # The offset should be rotated by the aligned orientation, not the raw one
        april_rot = Rotation.from_matrix(base_link_pose.rotation)
        april_rot_aligned = apply_rotation_alignment(april_rot, R_rot_align)
        offset_in_world = april_rot_aligned.as_matrix() @ tag_offset
        april_aligned = april_aligned_before_offset - offset_in_world

        # Convert AprilTag rotation matrix to quaternion
        april_quat_raw = april_rot.as_quat()  # [x, y, z, w]
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

    # Filter position data for correlation calculation
    april_positions_aligned_filtered = april_positions_aligned[inlier_mask]
    odom_positions_filtered = odom_positions[inlier_mask]

    # Compute correlation scores for position (X, Y, Z)
    corr_x = np.corrcoef(april_positions_aligned_filtered[:, 0], odom_positions_filtered[:, 0])[0, 1]
    corr_y = np.corrcoef(april_positions_aligned_filtered[:, 1], odom_positions_filtered[:, 1])[0, 1]
    corr_z = np.corrcoef(april_positions_aligned_filtered[:, 2], odom_positions_filtered[:, 2])[0, 1]

    # Filter quaternion data for rotation metrics
    april_quats_aligned_filtered = april_quats_aligned[inlier_mask]
    odom_quats_filtered = odom_quats[inlier_mask]

    # Compute INVARIANT metrics (coordinate-frame independent)
    # These are the metrics that actually matter for tracking quality

    # 1. Rotation speed magnitude (scalar - frame independent)
    april_rot_speed = []
    odom_rot_speed = []
    for i in range(1, len(april_quats_aligned_filtered)):
        r_april_prev = Rotation.from_quat(april_quats_aligned_filtered[i-1])
        r_april_curr = Rotation.from_quat(april_quats_aligned_filtered[i])
        r_odom_prev = Rotation.from_quat(odom_quats_filtered[i-1])
        r_odom_curr = Rotation.from_quat(odom_quats_filtered[i])

        # Scalar rotation magnitude (degrees per frame)
        april_rot_speed.append((r_april_curr * r_april_prev.inv()).magnitude() * 180 / np.pi)
        odom_rot_speed.append((r_odom_curr * r_odom_prev.inv()).magnitude() * 180 / np.pi)

    april_rot_speed = np.array(april_rot_speed)
    odom_rot_speed = np.array(odom_rot_speed)

    # 2. Rotation speed correlation (activity correlation - invariant)
    if len(april_rot_speed) > 2 and np.std(april_rot_speed) > 1e-6 and np.std(odom_rot_speed) > 1e-6:
        rot_speed_corr = np.corrcoef(april_rot_speed, odom_rot_speed)[0, 1]
    else:
        rot_speed_corr = np.nan

    # 3. Cumulative rotation (total degrees traveled)
    april_cumulative = np.sum(april_rot_speed)
    odom_cumulative = np.sum(odom_rot_speed)
    cumulative_ratio = april_cumulative / odom_cumulative if odom_cumulative > 0 else np.nan

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
    print("\nPosition correlation:")
    print(f"  X: {corr_x:.4f}")
    print(f"  Y: {corr_y:.4f}")
    print(f"  Z: {corr_z:.4f}")

    print("\n=== Orientation Comparison (Invariant Metrics) ===")
    print(f"Geodesic Error (total angular difference):")
    print(f"  Mean:            {mean_angle_err:.1f} deg")
    print(f"  Std:             {std_angle_err:.1f} deg")
    print(f"  90th percentile: {p90_angle:.1f} deg")
    print(f"  Max:             {max_angle_err:.1f} deg")

    print(f"\nRotation Speed (activity check):")
    print(f"  AprilTag total:  {april_cumulative:.1f} deg")
    print(f"  Odom total:      {odom_cumulative:.1f} deg")
    print(f"  Ratio:           {cumulative_ratio:.3f}" if not np.isnan(cumulative_ratio) else "  Ratio:           N/A")
    print(f"  Speed corr:      {rot_speed_corr:.4f}" if not np.isnan(rot_speed_corr) else "  Speed corr:      N/A")

    # Compute axis mapping quality from the rotation alignment matrix
    # The columns of R_rot_align represent AprilTag axes in Odom frame
    # Perfect alignment: each column is a unit vector along one axis (±1 in one component, 0 elsewhere)
    axis_names_short = ['Roll', 'Pitch', 'Yaw']

    # Compute alignment strength for each axis from R_rot_align
    alignment_strengths = []
    axis_mapping = []
    for i in range(3):
        # Find which odom axis this april axis maps to (column i of R_rot_align)
        col = R_rot_align[:, i]
        odom_axis_idx = np.argmax(np.abs(col))
        strength = np.abs(col[odom_axis_idx])
        sign = '+' if col[odom_axis_idx] > 0 else '-'
        alignment_strengths.append(strength)
        axis_mapping.append((odom_axis_idx, sign, strength))

    # Overall alignment quality (mean of axis strengths)
    # Perfect alignment = 1.0, completely off = lower values
    overall_alignment = np.mean(alignment_strengths)

    # Grade the alignment strength (based on rotation matrix column magnitudes)
    def grade_strength(strength):
        if strength >= 0.99:
            return 'A+'
        elif strength >= 0.95:
            return 'A'
        elif strength >= 0.90:
            return 'B'
        elif strength >= 0.80:
            return 'C'
        elif strength >= 0.70:
            return 'D'
        else:
            return 'F'

    # Print summary table
    print("\n" + "=" * 70)
    print("                        ALIGNMENT SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         POSITION METRICS                           │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Mean Error:     {mean_pos_err:6.1f} cm    │  Correlation X:  {corr_x:6.4f}       │")
    print(f"│  Std Error:      {std_pos_err:6.1f} cm    │  Correlation Y:  {corr_y:6.4f}       │")
    print(f"│  90th %ile:      {p90_pos:6.1f} cm    │  Correlation Z:  {corr_z:6.4f}       │")
    print(f"│  Max Error:      {max_pos_err:6.1f} cm    │  Frames:         {n_inliers:6d}         │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # Grade rotation speed correlation
    def grade_corr(corr):
        if np.isnan(corr):
            return 'N/A'
        elif corr >= 0.9:
            return 'A+'
        elif corr >= 0.8:
            return 'A'
        elif corr >= 0.6:
            return 'B'
        elif corr >= 0.4:
            return 'C'
        elif corr >= 0.2:
            return 'D'
        else:
            return 'F'

    # Grade cumulative ratio (should be close to 1.0)
    def grade_ratio(ratio):
        if np.isnan(ratio):
            return 'N/A'
        diff = abs(ratio - 1.0)
        if diff <= 0.05:
            return 'A+'
        elif diff <= 0.1:
            return 'A'
        elif diff <= 0.2:
            return 'B'
        elif diff <= 0.3:
            return 'C'
        elif diff <= 0.5:
            return 'D'
        else:
            return 'F'

    speed_corr_grade = grade_corr(rot_speed_corr)
    ratio_grade = grade_ratio(cumulative_ratio)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                 ORIENTATION METRICS (Invariant)                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Geodesic Error (angular difference between rotations):           │")
    print(f"│    Mean:       {mean_angle_err:6.1f} deg    90th %ile:   {p90_angle:6.1f} deg          │")
    print(f"│    Std:        {std_angle_err:6.1f} deg    Max:         {max_angle_err:6.1f} deg          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Rotation Activity (frame-independent):                           │")
    rot_speed_str = f"{rot_speed_corr:6.3f}" if not np.isnan(rot_speed_corr) else "   N/A"
    ratio_str = f"{cumulative_ratio:6.3f}" if not np.isnan(cumulative_ratio) else "   N/A"
    print(f"│    Speed Corr: {rot_speed_str}  ({speed_corr_grade:3})  │  Total AprilTag: {april_cumulative:7.1f} deg  │")
    print(f"│    Cumul Ratio:{ratio_str}  ({ratio_grade:3})  │  Total Odom:     {odom_cumulative:7.1f} deg  │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                   ROTATION AXIS ALIGNMENT (Kabsch)                 │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  Axis Mapping (from R_align rotation matrix):                      │")
    odom_axis_names = ['Odom X', 'Odom Y', 'Odom Z']
    for i, name in enumerate(axis_names_short):
        odom_idx, sign, strength = axis_mapping[i]
        grade = grade_strength(strength)
        bar_len = int(max(0, min(20, strength * 20)))
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"│  April {name:5} → {sign}{odom_axis_names[odom_idx]:6}  [{bar}] {strength:.3f} ({grade})  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    overall_grade = grade_strength(overall_alignment)
    overall_bar_len = int(max(0, min(20, overall_alignment * 20)))
    overall_bar = '█' * overall_bar_len + '░' * (20 - overall_bar_len)
    print(f"│  OVERALL:        [{overall_bar}] {overall_alignment:.3f} ({overall_grade})  │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # Interpretation
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         INTERPRETATION                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    if overall_alignment >= 0.95:
        print("│  ✓ Rotation alignment is excellent - axes map cleanly             │")
    elif overall_alignment >= 0.85:
        print("│  ~ Rotation alignment is good - minor axis mixing                 │")
    else:
        print("│  ✗ Rotation alignment has issues - check camera setup             │")

    if mean_pos_err < 10:
        print("│  ✓ Position accuracy is excellent (<10cm)                         │")
    elif mean_pos_err < 20:
        print("│  ~ Position accuracy is acceptable (10-20cm)                      │")
    else:
        print("│  ✗ Position accuracy needs improvement (>20cm)                    │")

    if mean_angle_err < 5:
        print("│  ✓ Orientation accuracy is excellent (<5°)                        │")
    elif mean_angle_err < 10:
        print("│  ~ Orientation accuracy is acceptable (5-10°)                     │")
    else:
        print("│  ✗ Orientation accuracy needs improvement (>10°)                  │")

    # Check rotation activity tracking (invariant metrics)
    if not np.isnan(rot_speed_corr) and rot_speed_corr >= 0.8:
        print("│  ✓ Rotation activity tracking is excellent (speed corr >= 0.8)     │")
    elif not np.isnan(rot_speed_corr) and rot_speed_corr >= 0.5:
        print("│  ~ Rotation activity tracking is acceptable (speed corr >= 0.5)    │")
    elif not np.isnan(rot_speed_corr):
        print("│  ✗ Rotation activity tracking needs work (speed corr < 0.5)        │")

    # Check yaw alignment specifically (most important for ground robots)
    yaw_strength = alignment_strengths[2]
    if yaw_strength >= 0.95:
        print("│  ✓ Yaw alignment is excellent (most important for ground robots)   │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print("=" * 70)

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
        'rot_speed_correlation': rot_speed_corr,  # Invariant rotation speed correlation
        'cumulative_rotation_april': april_cumulative,
        'cumulative_rotation_odom': odom_cumulative,
        'cumulative_ratio': cumulative_ratio,
    }
