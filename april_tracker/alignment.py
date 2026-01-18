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
) -> tuple[Rotation, np.ndarray]:
    """
    Compute rotation alignment between AprilTag and odometry orientations.

    Uses iterative optimization to find R_pre such that:
        R_odom ≈ R_offset @ R_pre @ R_april @ R_pre.T

    The R_pre matrix is found by searching over all 24 axis permutations,
    then refined using gradient-free optimization.

    Returns:
        R_offset: Residual rotation offset
        R_pre: 3x3 pre-multiplication matrix for similarity transform
    """
    from scipy.optimize import minimize

    april_rots = [Rotation.from_quat(q) for q in april_quats]
    odom_rots = [Rotation.from_quat(q) for q in odom_quats]

    def compute_rms_error(R_pre, R_offset=None):
        """Compute RMS angular error for given R_pre (and optionally R_offset)."""
        if R_offset is None:
            # Compute optimal offset for this R_pre
            relative_rots = []
            for r_april, r_odom in zip(april_rots, odom_rots):
                R_sim = R_pre @ r_april.as_matrix() @ R_pre.T
                r_sim = Rotation.from_matrix(R_sim)
                r_rel = r_odom * r_sim.inv()
                relative_rots.append(r_rel)

            quats = np.array([r.as_quat() for r in relative_rots])
            for i in range(1, len(quats)):
                if np.dot(quats[i], quats[0]) < 0:
                    quats[i] = -quats[i]
            mean_quat = np.mean(quats, axis=0)
            mean_quat = mean_quat / np.linalg.norm(mean_quat)
            R_offset = Rotation.from_quat(mean_quat)

        total_err = 0.0
        for r_april, r_odom in zip(april_rots, odom_rots):
            R_sim = R_pre @ r_april.as_matrix() @ R_pre.T
            r_sim = Rotation.from_matrix(R_sim)
            r_aligned = R_offset * r_sim
            r_diff = r_odom * r_aligned.inv()
            total_err += r_diff.magnitude() ** 2

        return np.sqrt(total_err / len(april_quats)), R_offset

    # Step 1: Try all 24 proper rotations (axis permutations)
    print("  Searching over 24 axis permutations...")
    best_perm_error = float('inf')
    best_P = np.eye(3)

    axis_permutations = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    ]
    sign_combos = [
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]

    for perm in axis_permutations:
        for signs in sign_combos:
            P = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                P[i, p] = s

            if np.linalg.det(P) < 0:
                continue

            rms, _ = compute_rms_error(P)
            if rms < best_perm_error:
                best_perm_error = rms
                best_P = P.copy()

    print(f"  Best axis permutation RMS error: {best_perm_error * 180 / np.pi:.2f} deg")
    print("  Best permutation matrix P:")
    print(f"    [{best_P[0, 0]:6.3f}, {best_P[0, 1]:6.3f}, {best_P[0, 2]:6.3f}]")
    print(f"    [{best_P[1, 0]:6.3f}, {best_P[1, 1]:6.3f}, {best_P[1, 2]:6.3f}]")
    print(f"    [{best_P[2, 0]:6.3f}, {best_P[2, 1]:6.3f}, {best_P[2, 2]:6.3f}]")

    # Step 2: Refine using continuous optimization
    # Parameterize R_pre as a rotation (3 parameters: axis-angle)
    print("  Refining with continuous optimization...")

    def objective(rotvec):
        R_pre = Rotation.from_rotvec(rotvec).as_matrix()
        rms, _ = compute_rms_error(R_pre)
        return rms

    # Start from best permutation (convert to axis-angle)
    init_rotvec = Rotation.from_matrix(best_P).as_rotvec()

    result = minimize(
        objective,
        init_rotvec,
        method='Powell',
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    R_pre_opt = Rotation.from_rotvec(result.x).as_matrix()
    rms_opt, R_offset_opt = compute_rms_error(R_pre_opt)

    print(f"  Optimized RMS error: {rms_opt * 180 / np.pi:.2f} deg")
    print("  Optimized R_pre:")
    print(f"    [{R_pre_opt[0, 0]:6.3f}, {R_pre_opt[0, 1]:6.3f}, {R_pre_opt[0, 2]:6.3f}]")
    print(f"    [{R_pre_opt[1, 0]:6.3f}, {R_pre_opt[1, 1]:6.3f}, {R_pre_opt[1, 2]:6.3f}]")
    print(f"    [{R_pre_opt[2, 0]:6.3f}, {R_pre_opt[2, 1]:6.3f}, {R_pre_opt[2, 2]:6.3f}]")
    print(f"  Residual offset (xyz): {R_offset_opt.as_euler('xyz', degrees=True)} deg")

    # Step 3: Also try simple left-multiply for comparison
    print("  Comparing with simple left-multiply...")
    relative_rotations = []
    for r_april, r_odom in zip(april_rots, odom_rots):
        r_left = r_odom * r_april.inv()
        relative_rotations.append(r_left)

    quats = np.array([r.as_quat() for r in relative_rotations])
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    mean_quat = np.mean(quats, axis=0)
    mean_quat = mean_quat / np.linalg.norm(mean_quat)
    R_left = Rotation.from_quat(mean_quat)

    total_left_err = 0.0
    for r_april, r_odom in zip(april_rots, odom_rots):
        r_aligned = R_left * r_april
        r_diff = r_odom * r_aligned.inv()
        total_left_err += r_diff.magnitude() ** 2
    rms_left = np.sqrt(total_left_err / len(april_quats)) * 180 / np.pi
    print(f"  Left-multiply RMS error: {rms_left:.2f} deg")

    # Choose the best approach
    rms_opt_deg = rms_opt * 180 / np.pi
    if rms_opt_deg < rms_left:
        print(f"  Using optimized similarity transform (better by {rms_left - rms_opt_deg:.2f} deg)")
        return R_offset_opt, R_pre_opt
    else:
        print(f"  Using left-multiply (better by {rms_opt_deg - rms_left:.2f} deg)")
        return R_left, np.eye(3)


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
