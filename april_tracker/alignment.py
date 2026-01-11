"""Alignment functions for matching AprilTag and odometry coordinate frames."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


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

        # Compute residuals for all points (XY only, Z is constant in odom)
        april_transformed = scale * (R @ april_positions.T).T + t
        residuals = np.linalg.norm(april_transformed[:, :2] - odom_positions[:, :2], axis=1)

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
