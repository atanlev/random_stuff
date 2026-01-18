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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RIGID rotation and translation (Scale=1.0) to align AprilTag coords to odometry coords.
    Uses Kabsch algorithm with iterative outlier removal.

    Returns:
        R: 3x3 rotation matrix (this IS the axis mapping)
        t: 3x1 translation vector
        inlier_mask: boolean array indicating inliers
    Such that: odom_pos ≈ R @ april_pos + t
    """
    inlier_mask = np.ones(len(april_positions), dtype=bool)

    R = np.eye(3)
    t = np.zeros(3)

    for iteration in range(max_iterations):
        april_inliers = april_positions[inlier_mask]
        odom_inliers = odom_positions[inlier_mask]

        if len(april_inliers) < 3:
            print("Warning: Not enough inliers for alignment.")
            break

        # 1. Centroids
        april_mean = np.mean(april_inliers, axis=0)
        odom_mean = np.mean(odom_inliers, axis=0)

        # 2. Center the points
        april_centered = april_inliers - april_mean
        odom_centered = odom_inliers - odom_mean

        # 3. Covariance Matrix
        H = april_centered.T @ odom_centered

        # 4. SVD for Rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 5. Handle Reflection (ensure proper rotation, det(R)=1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 6. Compute Translation (RIGID: scale = 1.0)
        t = odom_mean - R @ april_mean

        # 7. Compute residuals for all points
        april_transformed = (R @ april_positions.T).T + t
        residuals = np.linalg.norm(april_transformed - odom_positions, axis=1)

        # 8. Update inlier mask
        new_inlier_mask = residuals < outlier_threshold
        n_outliers = np.sum(~new_inlier_mask)

        if iteration > 0:
            print(f"  Iteration {iteration + 1}: {n_outliers} outliers (threshold={outlier_threshold:.2f}m)")

        # Stop if no change
        if np.array_equal(new_inlier_mask, inlier_mask):
            break

        inlier_mask = new_inlier_mask

    return R, t, inlier_mask


def compute_rotation_alignment(
    april_quats: list[np.ndarray],
    odom_quats: list[np.ndarray],
) -> np.ndarray:
    """
    Compute rotation alignment between AprilTag and odometry orientations.

    Uses Kabsch algorithm on rotation matrices to find the optimal
    rotation R such that: R_odom ≈ R_align @ R_april

    Returns:
        R_align: 3x3 rotation matrix for alignment
    """
    april_rots = [Rotation.from_quat(q) for q in april_quats]
    odom_rots = [Rotation.from_quat(q) for q in odom_quats]

    # Convert to rotation matrices
    april_mats = np.array([r.as_matrix() for r in april_rots])
    odom_mats = np.array([r.as_matrix() for r in odom_rots])

    # Use Kabsch on the rotation matrices
    # We want to find R_align such that R_odom ≈ R_align @ R_april
    # This is equivalent to finding R_align that minimizes ||R_odom - R_align @ R_april||
    #
    # Using the Frobenius norm and averaging over all samples:
    # H = sum(R_odom @ R_april.T) = sum of cross-covariance
    H = np.zeros((3, 3))
    for r_april, r_odom in zip(april_mats, odom_mats):
        H += r_odom @ r_april.T

    # SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(H)
    R_align = U @ Vt

    # Handle reflection
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = U @ Vt

    # Print axis mapping interpretation
    print("\n  Rotation alignment (R_align @ R_april = R_odom):")
    _print_axis_mapping(R_align)

    # Compute alignment error
    errors = []
    for r_april, r_odom in zip(april_rots, odom_rots):
        r_aligned = Rotation.from_matrix(R_align) * r_april
        r_diff = r_odom * r_aligned.inv()
        errors.append(r_diff.magnitude() * 180 / np.pi)

    errors = np.array(errors)
    print(f"\n  Alignment error: mean={np.mean(errors):.2f}°, std={np.std(errors):.2f}°, max={np.max(errors):.2f}°")

    return R_align


def _print_axis_mapping(R: np.ndarray) -> None:
    """
    Interprets the rotation matrix to show which axis maps to which.
    R @ v_april = v_odom
    The columns of R represent the AprilTag axes expressed in Odom frame.
    """
    axes_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']

    print("  Axis Mapping (AprilTag -> Odometry):")

    # Check columns of R (AprilTag axes -> Odom frame)
    for i in range(3):
        # Find which row (Odom axis) has the strongest magnitude
        odom_axis_idx = np.argmax(np.abs(R[:, i]))
        sign = np.sign(R[odom_axis_idx, i])
        strength = np.abs(R[odom_axis_idx, i])
        sign_str = "+" if sign > 0 else "-"

        print(f"    April {axes_names[i]:12} -> {sign_str}Odom {axes_names[odom_axis_idx]:12} (strength: {strength:.3f})")

    # Check determinant
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-3:
        print(f"  WARNING: Matrix is not a proper rotation (det={det:.3f})")


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
