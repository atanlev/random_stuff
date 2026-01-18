"""Visualization functions for AprilTag vs odometry comparison."""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.spatial.transform import Rotation

from .processing import apply_rotation_alignment

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
from .config import DEBUG_AXIS_FLIP_X, DEBUG_AXIS_FLIP_Y, DEBUG_AXIS_FLIP_Z
from .processing import apply_tag_to_baselink_offset

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
    R_rot_align: tuple,
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
            # Apply offset to get base_link position from AprilTag position (in robot frame)
            april_robot = apply_tag_to_baselink_offset(
                base_link_pose.position, base_link_pose.rotation
            )

            # Transform odom position back to robot frame (same as april)
            odom_pos = odom['position']
            odom_robot = R_inv @ (odom_pos - t) * scale_inv

            # Transform from robot frame to camera frame for projection
            # Robot frame (from tracker.py): X=forward (toward camera), Y=left, Z=up
            # BUT the position is the tag's position as seen from camera,
            # so Robot X is negative (tag is in front of camera, pointing away)
            # Camera frame: X=right, Y=down, Z=forward
            #
            # The tracker outputs positions where:
            #   Robot X = -Camera Z (forward in robot = backward in camera depth)
            #   Robot Y = Camera X (left in robot = right in camera)
            #   Robot Z = -Camera Y (up in robot = down in camera)
            # So to go back: Camera = inverse of that transform
            #   Camera X = Robot Y
            #   Camera Y = -Robot Z
            #   Camera Z = -Robot X
            R_robot_to_cam = np.array([
                [0,  1,  0],   # Camera X from Robot Y
                [0,  0, -1],   # Camera Y from -Robot Z
                [-1, 0,  0],   # Camera Z from -Robot X
            ])

            april_camera = R_robot_to_cam @ april_robot
            odom_camera = R_robot_to_cam @ odom_robot

            # Project to image
            cam_mat = tracker.camera_matrix
            if april_camera[2] > 0:  # In front of camera
                april_x = int(cam_mat[0, 0] * april_camera[0] / april_camera[2] + cam_mat[0, 2])
                april_y = int(cam_mat[1, 1] * april_camera[1] / april_camera[2] + cam_mat[1, 2])

                if odom_camera[2] > 0:  # Check odom is in front too
                    odom_x = int(cam_mat[0, 0] * odom_camera[0] / odom_camera[2] + cam_mat[0, 2])
                    odom_y = int(cam_mat[1, 1] * odom_camera[1] / odom_camera[2] + cam_mat[1, 2])
                else:
                    odom_x = april_x
                    odom_y = april_y

                # Get quaternions for orientation axes
                # AprilTag quaternion (in robot frame) - apply rotation alignment
                april_rot_raw = Rotation.from_matrix(base_link_pose.rotation)
                april_rot_aligned = apply_rotation_alignment(april_rot_raw, R_rot_align)

                # Odometry quaternion (in world/odom frame)
                odom_quat = odom['quaternion']
                odom_rot_world = Rotation.from_quat(odom_quat)

                # For visualization, we want both in the same frame
                # The aligned AprilTag rotation is now in world frame (like odometry)
                # Transform both to robot frame for visualization
                R_world_to_robot = Rotation.from_matrix(R.T)
                april_rot_robot = R_world_to_robot * april_rot_aligned
                odom_rot_robot = R_world_to_robot * odom_rot_world

                # Transform from robot frame to OpenCV camera frame for visualization
                # (Same transform as above for positions)
                R_robot_to_cam = np.array([
                    [0,  1,  0],   # Camera X from Robot Y
                    [0,  0, -1],   # Camera Y from -Robot Z
                    [-1, 0,  0],   # Camera Z from -Robot X
                ])

                # Draw coordinate axes
                axis_length = 40

                # Axis colors: X=Red, Y=Green, Z=Blue (RGB convention)
                axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR for OpenCV
                axis_dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

                # For AprilTag point (blue circle): draw AprilTag axes
                for i, (axis_dir, color) in enumerate(zip(axis_dirs, axis_colors)):
                    # Apply rotation in robot frame
                    axis_robot = april_rot_robot.apply(axis_dir)

                    # Apply debug flips
                    flip_signs = np.array([
                        -1.0 if DEBUG_AXIS_FLIP_X else 1.0,
                        -1.0 if DEBUG_AXIS_FLIP_Y else 1.0,
                        -1.0 if DEBUG_AXIS_FLIP_Z else 1.0
                    ])
                    axis_robot = axis_robot * flip_signs

                    # Scale axis for visualization (in meters in robot frame)
                    axis_robot_scaled = axis_robot * 0.1  # 10cm axis length
                    # Add to the position to get endpoint in robot frame
                    axis_endpoint_robot = april_robot + axis_robot_scaled
                    # Transform to OpenCV camera frame
                    axis_endpoint_camera = R_robot_to_cam @ axis_endpoint_robot
                    # Project to image using pinhole model
                    if axis_endpoint_camera[2] > 0.01:  # Only project if in front
                        axis_endpoint_x = int(cam_mat[0, 0] * axis_endpoint_camera[0] / axis_endpoint_camera[2] + cam_mat[0, 2])
                        axis_endpoint_y = int(cam_mat[1, 1] * axis_endpoint_camera[1] / axis_endpoint_camera[2] + cam_mat[1, 2])
                        cv2.arrowedLine(frame, (april_x, april_y),
                                       (axis_endpoint_x, axis_endpoint_y),
                                       color, 2, tipLength=0.3)

                # For Odometry point (red circle): draw odometry axes
                for i, (axis_dir, color) in enumerate(zip(axis_dirs, axis_colors)):
                    # Apply rotation in robot frame (after transformation from world)
                    axis_robot = odom_rot_robot.apply(axis_dir)

                    # Apply debug flips
                    flip_signs = np.array([
                        -1.0 if DEBUG_AXIS_FLIP_X else 1.0,
                        -1.0 if DEBUG_AXIS_FLIP_Y else 1.0,
                        -1.0 if DEBUG_AXIS_FLIP_Z else 1.0
                    ])
                    axis_robot = axis_robot * flip_signs

                    # Scale axis for visualization (in meters in robot frame)
                    axis_robot_scaled = axis_robot * 0.1  # 10cm axis length
                    # Add to the position to get endpoint in robot frame
                    axis_endpoint_robot = odom_robot + axis_robot_scaled
                    # Transform to OpenCV camera frame
                    axis_endpoint_camera = R_robot_to_cam @ axis_endpoint_robot
                    # Project to image using pinhole model
                    if axis_endpoint_camera[2] > 0.01:  # Only project if in front
                        axis_endpoint_x = int(cam_mat[0, 0] * axis_endpoint_camera[0] / axis_endpoint_camera[2] + cam_mat[0, 2])
                        axis_endpoint_y = int(cam_mat[1, 1] * axis_endpoint_camera[1] / axis_endpoint_camera[2] + cam_mat[1, 2])
                        cv2.arrowedLine(frame, (odom_x, odom_y),
                                       (axis_endpoint_x, axis_endpoint_y),
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
                april_aligned = scale * (R @ april_robot) + t
                pos_error = np.linalg.norm(april_aligned - odom_pos) * 100  # cm

                # Compute angle error (using aligned rotations in world frame)
                r_diff = odom_rot_world.inv() * april_rot_aligned
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

        # Try to display frame, but don't fail if display is unavailable
        try:
            cv2.imshow('AprilTag vs Odometry', frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        except cv2.error:
            # Display not available, just continue processing frames
            pass

        frame_idx += 1

    if writer:
        writer.release()
        print(f"Saved visualization video to: {output_path}")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
