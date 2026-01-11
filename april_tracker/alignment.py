"""Alignment functions for matching AprilTag and odometry coordinate frames."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d


def compute_alignment_transform(
    april_positions: np.ndarray,
    odom_positions: np.ndarray,
    outlier_threshold: float = 0.5,
    max_iterations: int = 3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute rotation, scale, and translation to align AprilTag coords to odometry coords.
    Uses Kabsch algorithm with iterative outlier removal.

    Returns:
        R: 3x3 rotation matrix
        scale: scale factor
        t: 3x1 translation vector
        inlier_mask: boolean array indicating inliers
    Such that: odom_pos ≈ scale * (R @ april_pos) + t
    """
    inlier_mask = np.ones(len(april_positions), dtype=bool)

    for iteration in range(max_iterations):
        april_inliers = april_positions[inlier_mask]
        odom_inliers = odom_positions[inlier_mask]

        april_mean = np.mean(april_inliers, axis=0)
        odom_mean = np.mean(odom_inliers, axis=0)

        april_centered = april_inliers - april_mean
        odom_centered = odom_inliers - odom_mean

        # Compute covariance matrix
        H = april_centered.T @ odom_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute scale
        april_rotated = (R @ april_centered.T).T
        scale = np.linalg.norm(odom_centered) / np.linalg.norm(april_rotated)

        # Compute translation
        t = odom_mean - scale * (R @ april_mean)

        # Compute residuals for all points (XYZ)
        april_transformed = scale * (R @ april_positions.T).T + t
        residuals = np.linalg.norm(april_transformed - odom_positions, axis=1)

        # Update inlier mask
        new_inlier_mask = residuals < outlier_threshold
        n_outliers = np.sum(~new_inlier_mask)

        if iteration > 0:
            print(f"  Iteration {iteration + 1}: {n_outliers} outliers (threshold={outlier_threshold:.2f}m)")

        # Stop if no change
        if np.array_equal(new_inlier_mask, inlier_mask):
            break

        inlier_mask = new_inlier_mask

    return R, scale, t, inlier_mask


def compute_rotation_alignment(
    april_quats: list[np.ndarray],
    odom_quats: list[np.ndarray],
) -> Rotation:
    """
    Compute rotation alignment between AprilTag and odometry orientations.
    Uses average of relative rotations from stationary frames.

    Returns:
        R_rot_align: Rotation to apply to AprilTag quaternions to align with odometry
    """
    # Compute relative rotation for each pair: R_align = R_odom * R_april^-1
    relative_rotations = []
    for april_q, odom_q in zip(april_quats, odom_quats):
        r_april = Rotation.from_quat(april_q)
        r_odom = Rotation.from_quat(odom_q)
        r_rel = r_odom * r_april.inv()
        relative_rotations.append(r_rel)

    # Average the relative rotations using mean of quaternions
    quats = np.array([r.as_quat() for r in relative_rotations])

    # Handle quaternion sign ambiguity - flip quats to same hemisphere as first
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    mean_quat = np.mean(quats, axis=0)
    mean_quat = mean_quat / np.linalg.norm(mean_quat)  # normalize

    return Rotation.from_quat(mean_quat)


def find_time_offset(
    april_timestamps: np.ndarray,
    april_positions: np.ndarray,
    odom_timestamps: np.ndarray,
    odom_positions: np.ndarray,
    search_range_ms: float = 100.0,
    step_ms: float = 2.0,
) -> tuple[float, float]:
    """
    Find optimal time offset between AprilTag and odometry using cross-correlation.

    The odometry timestamp is when the TF was requested, but the actual pose
    corresponds to a point slightly in the past. We find the offset that minimizes
    position error after applying Kabsch alignment.

    Args:
        april_timestamps: Timestamps for AprilTag positions
        april_positions: AprilTag positions (Nx3)
        odom_timestamps: Timestamps for odometry positions
        odom_positions: Odometry positions (Mx3)
        search_range_ms: Search range in milliseconds (searches +/- this range)
        step_ms: Step size in milliseconds

    Returns:
        best_offset_s: Optimal offset in seconds (positive = odom is delayed)
        min_error: Mean position error at optimal offset
    """
    # Convert to seconds
    search_range_s = search_range_ms / 1000.0
    step_s = step_ms / 1000.0

    # Create interpolator for odometry positions
    odom_interp_x = interp1d(odom_timestamps, odom_positions[:, 0],
                              kind='linear', bounds_error=False, fill_value=np.nan)
    odom_interp_y = interp1d(odom_timestamps, odom_positions[:, 1],
                              kind='linear', bounds_error=False, fill_value=np.nan)
    odom_interp_z = interp1d(odom_timestamps, odom_positions[:, 2],
                              kind='linear', bounds_error=False, fill_value=np.nan)

    offsets = np.arange(-search_range_s, search_range_s + step_s, step_s)
    errors = []

    for offset in offsets:
        # Shift april timestamps by offset (if odom is delayed, we look earlier)
        shifted_timestamps = april_timestamps - offset

        # Interpolate odometry at shifted timestamps
        odom_x = odom_interp_x(shifted_timestamps)
        odom_y = odom_interp_y(shifted_timestamps)
        odom_z = odom_interp_z(shifted_timestamps)

        # Stack and find valid (non-nan) points
        odom_interp = np.column_stack([odom_x, odom_y, odom_z])
        valid_mask = ~np.isnan(odom_interp).any(axis=1)

        if np.sum(valid_mask) < 10:
            errors.append(np.inf)
            continue

        april_valid = april_positions[valid_mask]
        odom_valid = odom_interp[valid_mask]

        # Use Kabsch algorithm to align, then compute residuals
        # This properly handles rotation between coordinate frames
        april_mean = np.mean(april_valid, axis=0)
        odom_mean = np.mean(odom_valid, axis=0)

        april_centered = april_valid - april_mean
        odom_centered = odom_valid - odom_mean

        # SVD for rotation
        H = april_centered.T @ odom_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute scale
        april_rotated = (R @ april_centered.T).T
        scale = np.linalg.norm(odom_centered) / (np.linalg.norm(april_rotated) + 1e-10)

        # Transform april to odom frame
        t = odom_mean - scale * (R @ april_mean)
        april_transformed = scale * (R @ april_valid.T).T + t

        # Compute residuals
        residuals = np.linalg.norm(april_transformed - odom_valid, axis=1)
        mean_error = np.mean(residuals)
        errors.append(mean_error)

    errors = np.array(errors)
    best_idx = np.argmin(errors)
    best_offset_s = offsets[best_idx]
    min_error = errors[best_idx]

    # Print a few sample offsets to show the error landscape
    sample_offsets = [0, -50, -100, -150, -200, 50, 100]
    print("  Time offset search results:")
    for sample_ms in sample_offsets:
        sample_s = sample_ms / 1000.0
        idx = np.argmin(np.abs(offsets - sample_s))
        if idx < len(errors) and not np.isinf(errors[idx]):
            print(f"    {sample_ms:+4d}ms: {errors[idx]:.4f}m")

    if best_offset_s > 0:
        raise ValueError(f"Best offset is positive: {best_offset_s * 1000:.1f} ms (odom delayed)")
    return best_offset_s, min_error
