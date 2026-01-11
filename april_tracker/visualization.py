"""Visualization functions for AprilTag vs odometry comparison."""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.spatial.transform import Rotation

import os
import sys

# Suppress Qt threading warnings
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false;qt.*=false'
os.environ['QT_DEBUG_PLUGINS'] = '0'

# Redirect stderr temporarily during cv2 operations to suppress Qt warnings
class SuppressQtWarnings:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._stderr

from .tracker_data_types import FrameResult
from .tracker import AprilTagTracker
from .config import APRILTAG_TO_BASELINK_OFFSET


def apply_tag_to_baselink_offset(april_pos: np.ndarray, april_rot: np.ndarray) -> np.ndarray:
    """Transform AprilTag position to base_link position using the known offset."""
    offset_in_ref = april_rot @ APRILTAG_TO_BASELINK_OFFSET
    return april_pos - offset_in_ref

def plot_comparison(comparison: dict):
    """Plot AprilTag vs odometry position and orientation comparison."""
    april_pos = comparison['april_positions']
    odom_pos = comparison['odom_positions']
    april_quats = comparison['april_quats_aligned']
    odom_quats = comparison['odom_quats']
    timestamps = comparison['timestamps']
    pos_errors = comparison['pos_error_norms']

    # Normalize timestamps to start at 0
    t = timestamps - timestamps[0]

    # Convert quaternions to euler angles (in degrees)
    april_euler = np.array([Rotation.from_quat(q).as_euler('xyz', degrees=True) for q in april_quats])
    odom_euler = np.array([Rotation.from_quat(q).as_euler('xyz', degrees=True) for q in odom_quats])

    # Compute euler angle errors (handle wraparound at +/-180)
    euler_errors = april_euler - odom_euler
    # Wrap to [-180, 180]
    euler_errors = np.mod(euler_errors + 180, 360) - 180

    # Position plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Position Comparison', fontsize=14)

    # X position over time
    axes1[0, 0].plot(t, april_pos[:, 0], 'b-', label='AprilTag', alpha=0.7)
    axes1[0, 0].plot(t, odom_pos[:, 0], 'r--', label='Odometry', alpha=0.7)
    axes1[0, 0].set_xlabel('Time (s)')
    axes1[0, 0].set_ylabel('X (m)')
    axes1[0, 0].set_title('X Position')
    axes1[0, 0].legend()
    axes1[0, 0].grid(True)

    # Y position over time
    axes1[0, 1].plot(t, april_pos[:, 1], 'b-', label='AprilTag', alpha=0.7)
    axes1[0, 1].plot(t, odom_pos[:, 1], 'r--', label='Odometry', alpha=0.7)
    axes1[0, 1].set_xlabel('Time (s)')
    axes1[0, 1].set_ylabel('Y (m)')
    axes1[0, 1].set_title('Y Position')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True)

    # Z position over time
    axes1[1, 0].plot(t, april_pos[:, 2], 'b-', label='AprilTag', alpha=0.7)
    axes1[1, 0].plot(t, odom_pos[:, 2], 'r--', label='Odometry', alpha=0.7)
    axes1[1, 0].set_xlabel('Time (s)')
    axes1[1, 0].set_ylabel('Z (m)')
    axes1[1, 0].set_title('Z Position')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True)

    # Position error over time
    axes1[1, 1].plot(t, pos_errors * 100, 'g-', alpha=0.7)  # Convert to cm
    axes1[1, 1].set_xlabel('Time (s)')
    axes1[1, 1].set_ylabel('Error (cm)')
    axes1[1, 1].set_title(f'Position Error (mean={np.mean(pos_errors)*100:.1f}cm)')
    axes1[1, 1].grid(True)

    plt.tight_layout()

    # Orientation plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Orientation Comparison (Euler XYZ)', fontsize=14)

    # Roll (X rotation) over time
    axes2[0, 0].plot(t, april_euler[:, 0], 'b-', label='AprilTag', alpha=0.7)
    axes2[0, 0].plot(t, odom_euler[:, 0], 'r--', label='Odometry', alpha=0.7)
    axes2[0, 0].set_xlabel('Time (s)')
    axes2[0, 0].set_ylabel('Roll (deg)')
    axes2[0, 0].set_title('Roll (X rotation)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)

    # Pitch (Y rotation) over time
    axes2[0, 1].plot(t, april_euler[:, 1], 'b-', label='AprilTag', alpha=0.7)
    axes2[0, 1].plot(t, odom_euler[:, 1], 'r--', label='Odometry', alpha=0.7)
    axes2[0, 1].set_xlabel('Time (s)')
    axes2[0, 1].set_ylabel('Pitch (deg)')
    axes2[0, 1].set_title('Pitch (Y rotation)')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True)

    # Yaw (Z rotation) over time
    axes2[1, 0].plot(t, april_euler[:, 2], 'b-', label='AprilTag', alpha=0.7)
    axes2[1, 0].plot(t, odom_euler[:, 2], 'r--', label='Odometry', alpha=0.7)
    axes2[1, 0].set_xlabel('Time (s)')
    axes2[1, 0].set_ylabel('Yaw (deg)')
    axes2[1, 0].set_title('Yaw (Z rotation)')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True)

    # Euler angle errors over time
    axes2[1, 1].plot(t, euler_errors[:, 0], 'r-', label='Roll err', alpha=0.7)
    axes2[1, 1].plot(t, euler_errors[:, 1], 'g-', label='Pitch err', alpha=0.7)
    axes2[1, 1].plot(t, euler_errors[:, 2], 'b-', label='Yaw err', alpha=0.7)
    axes2[1, 1].set_xlabel('Time (s)')
    axes2[1, 1].set_ylabel('Error (deg)')
    axes2[1, 1].set_title(f'Euler Angle Errors (RMS: R={np.sqrt(np.mean(euler_errors[:, 0]**2)):.1f}, '
                          f'P={np.sqrt(np.mean(euler_errors[:, 1]**2)):.1f}, '
                          f'Y={np.sqrt(np.mean(euler_errors[:, 2]**2)):.1f} deg)')
    axes2[1, 1].legend()
    axes2[1, 1].grid(True)

    plt.tight_layout()

    # 2D trajectory plot (X-Y plane)
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.plot(april_pos[:, 0], april_pos[:, 1], 'b-', label='AprilTag', alpha=0.7)
    ax3.plot(odom_pos[:, 0], odom_pos[:, 1], 'r--', label='Odometry', alpha=0.7)
    ax3.scatter(april_pos[0, 0], april_pos[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax3.scatter(odom_pos[0, 0], odom_pos[0, 1], c='red', s=100, marker='o', zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Trajectory Comparison')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    plt.show()


def visualize_on_frames(
    zed_frames: list[dict],
    april_results: list[FrameResult],
    matched_odom: list[dict],
    tracker: AprilTagTracker,
    R: np.ndarray,
    scale: float,
    t: np.ndarray,
    R_rot_align: Rotation,
    output_path: Optional[str] = None,
):
    """
    Visualize AprilTag detection and odometry positions projected onto RGB frames.

    Blue circle: AprilTag detected position with odometry orientation arrow
    Red circle: Odometry position with AprilTag orientation arrow
    Green line: Error between them
    """
    # To project odometry back to image, we need the inverse transform
    # april_aligned = scale * (R @ april_raw) + t  =>  odom frame
    # To go from odom to april_raw: april_raw = R.T @ (odom - t) / scale
    R_inv = R.T
    scale_inv = 1.0 / scale

    height, width = zed_frames[0]['frame'].shape[:2]
    fps = 30

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nVisualizing frames - press 'q' to quit, 'space' to pause")

    paused = False
    frame_idx = 0

    for result, odom, frame_data in zip(april_results, matched_odom, zed_frames):
        frame = frame_data['frame'].copy()

        # Get AprilTag position if detected
        base_link_pose = result.poses.get('base_link')

        if base_link_pose is not None and odom is not None:
            # Apply offset to get base_link position from AprilTag position
            april_raw = apply_tag_to_baselink_offset(
                base_link_pose.position, base_link_pose.rotation
            )

            # Transform odom position back to AprilTag reference frame
            odom_pos = odom['position']
            odom_in_april_frame = R_inv @ (odom_pos - t) * scale_inv

            # Project both to image using tracker's reference pose
            ref_pose = tracker.reference_pose

            # Base_link position (from AprilTag with offset) in camera frame
            april_camera = ref_pose.rotation @ april_raw + ref_pose.position
            # Odom position in camera frame
            odom_camera = ref_pose.rotation @ odom_in_april_frame + ref_pose.position

            # Project to image
            cam_mat = tracker.camera_matrix
            if april_camera[2] > 0:  # In front of camera
                april_x = int(cam_mat[0, 0] * april_camera[0] / april_camera[2] + cam_mat[0, 2])
                april_y = int(cam_mat[1, 1] * april_camera[1] / april_camera[2] + cam_mat[1, 2])

                odom_x = int(cam_mat[0, 0] * odom_camera[0] / odom_camera[2] + cam_mat[0, 2])
                odom_y = int(cam_mat[1, 1] * odom_camera[1] / odom_camera[2] + cam_mat[1, 2])

                # Get quaternions for orientation axes
                # AprilTag quaternion (raw, then aligned)
                april_rot = Rotation.from_matrix(base_link_pose.rotation)
                april_rot_aligned = R_rot_align * april_rot
                # Odometry quaternion
                odom_quat = odom['quaternion']
                odom_rot = Rotation.from_quat(odom_quat)

                # Draw coordinate axes (SWAPPED: april point gets odom axes, odom point gets april axes)
                axis_length = 40

                # Axis colors: X=Red, Y=Green, Z=Blue (RGB convention)
                axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR for OpenCV
                axis_dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

                # For AprilTag point (blue circle): draw odometry coordinate axes
                for axis_dir, color in zip(axis_dirs, axis_colors):
                    odom_axis = odom_rot.apply(axis_dir)
                    dx = int(axis_length * odom_axis[0])
                    dy = int(-axis_length * odom_axis[1])  # flip y for image coords
                    cv2.arrowedLine(frame, (april_x, april_y),
                                   (april_x + dx, april_y + dy),
                                   color, 2, tipLength=0.3)

                # For Odometry point (red circle): draw AprilTag coordinate axes
                for axis_dir, color in zip(axis_dirs, axis_colors):
                    april_axis = april_rot_aligned.apply(axis_dir)
                    dx = int(axis_length * april_axis[0])
                    dy = int(-axis_length * april_axis[1])
                    cv2.arrowedLine(frame, (odom_x, odom_y),
                                   (odom_x + dx, odom_y + dy),
                                   color, 2, tipLength=0.3)

                # Draw error line (green)
                cv2.line(frame, (april_x, april_y), (odom_x, odom_y), (0, 255, 0), 2)

                # Draw AprilTag position (blue circle)
                cv2.circle(frame, (april_x, april_y), 12, (255, 0, 0), -1)
                cv2.circle(frame, (april_x, april_y), 14, (0, 0, 0), 2)

                # Draw Odometry position (red circle)
                cv2.circle(frame, (odom_x, odom_y), 12, (0, 0, 255), -1)
                cv2.circle(frame, (odom_x, odom_y), 14, (0, 0, 0), 2)

                # Compute and display errors (XYZ)
                april_aligned = scale * (R @ april_raw) + t
                pos_error = np.linalg.norm(april_aligned - odom_pos) * 100  # cm

                # Compute angle error
                r_diff = odom_rot.inv() * april_rot_aligned
                angle_error = np.abs(r_diff.magnitude()) * 180 / np.pi

                cv2.putText(frame, f"Pos err: {pos_error:.1f}cm  Ang err: {angle_error:.1f}deg", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw legend
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Position legend (top right)
        legend_x = width - 160
        cv2.circle(frame, (legend_x, 25), 8, (255, 0, 0), -1)
        cv2.putText(frame, "AprilTag", (legend_x + 15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (legend_x, 50), 8, (0, 0, 255), -1)
        cv2.putText(frame, "Odometry", (legend_x + 15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Axis legend (below position legend)
        cv2.putText(frame, "Axes:", (legend_x, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # X axis (red)
        cv2.arrowedLine(frame, (legend_x, 95), (legend_x + 25, 95), (0, 0, 255), 2, tipLength=0.4)
        cv2.putText(frame, "X", (legend_x + 30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Y axis (green)
        cv2.arrowedLine(frame, (legend_x + 50, 95), (legend_x + 75, 95), (0, 255, 0), 2, tipLength=0.4)
        cv2.putText(frame, "Y", (legend_x + 80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Z axis (blue)
        cv2.arrowedLine(frame, (legend_x + 100, 95), (legend_x + 125, 95), (255, 0, 0), 2, tipLength=0.4)
        cv2.putText(frame, "Z", (legend_x + 130, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if writer:
            writer.write(frame)

        with SuppressQtWarnings():
            cv2.imshow('AprilTag vs Odometry', frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

        frame_idx += 1

    if writer:
        writer.release()
        print(f"Saved visualization video to: {output_path}")

    with SuppressQtWarnings():
        cv2.destroyAllWindows()
