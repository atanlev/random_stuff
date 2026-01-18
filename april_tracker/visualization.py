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

    # Orientation plots - using meaningful representations instead of raw Euler angles
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Orientation Comparison', fontsize=14)

    # 1. Rotation error over time (angular difference - the actual tracking quality)
    angle_errors = []
    for april_q, odom_q in zip(april_quats, odom_quats):
        r_april = Rotation.from_quat(april_q)
        r_odom = Rotation.from_quat(odom_q)
        r_diff = r_odom.inv() * r_april
        angle_errors.append(np.abs(r_diff.magnitude()) * 180 / np.pi)
    angle_errors = np.array(angle_errors)

    axes2[0, 0].plot(t, angle_errors, 'g-', alpha=0.7)
    axes2[0, 0].axhline(y=np.mean(angle_errors), color='r', linestyle='--',
                        label=f'Mean: {np.mean(angle_errors):.1f}°')
    axes2[0, 0].set_xlabel('Time (s)')
    axes2[0, 0].set_ylabel('Angular Error (deg)')
    axes2[0, 0].set_title('Rotation Error (geodesic distance)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)

    # 2. Angular velocity comparison (frame-to-frame rotation changes)
    # This shows if tracking follows rotational motion correctly
    april_ang_vel = []
    odom_ang_vel = []
    for i in range(1, len(april_quats)):
        r_april_prev = Rotation.from_quat(april_quats[i-1])
        r_april_curr = Rotation.from_quat(april_quats[i])
        r_odom_prev = Rotation.from_quat(odom_quats[i-1])
        r_odom_curr = Rotation.from_quat(odom_quats[i])

        # Angular change magnitude (degrees)
        april_delta = (r_april_curr * r_april_prev.inv()).magnitude() * 180 / np.pi
        odom_delta = (r_odom_curr * r_odom_prev.inv()).magnitude() * 180 / np.pi
        april_ang_vel.append(april_delta)
        odom_ang_vel.append(odom_delta)

    april_ang_vel = np.array(april_ang_vel)
    odom_ang_vel = np.array(odom_ang_vel)
    t_vel = t[1:]  # One less point for velocities

    axes2[0, 1].plot(t_vel, april_ang_vel, 'b-', label='AprilTag', alpha=0.7)
    axes2[0, 1].plot(t_vel, odom_ang_vel, 'r--', label='Odometry', alpha=0.7)
    axes2[0, 1].set_xlabel('Time (s)')
    axes2[0, 1].set_ylabel('Angular Change (deg/frame)')
    axes2[0, 1].set_title('Frame-to-Frame Rotation Magnitude')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True)

    # 3. Rotation speed scatter plot (invariant activity correlation)
    # If points cluster along diagonal, tracking follows motion correctly
    # This is coordinate-frame independent
    axes2[1, 0].scatter(odom_ang_vel, april_ang_vel, alpha=0.5, s=10)
    # Add diagonal reference line
    max_vel = max(np.max(april_ang_vel), np.max(odom_ang_vel))
    axes2[1, 0].plot([0, max_vel], [0, max_vel], 'r--', alpha=0.5, label='Perfect match')
    # Compute correlation
    if len(april_ang_vel) > 2:
        corr = np.corrcoef(april_ang_vel, odom_ang_vel)[0, 1]
        axes2[1, 0].set_title(f'Activity Correlation (r={corr:.3f})')
    else:
        axes2[1, 0].set_title('Activity Correlation')
    axes2[1, 0].set_xlabel('Odom Rotation Speed (deg/frame)')
    axes2[1, 0].set_ylabel('AprilTag Rotation Speed (deg/frame)')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True)
    axes2[1, 0].set_aspect('equal', adjustable='box')

    # 4. Cumulative rotation (total degrees traveled)
    # This is invariant - shows if both systems "traveled" the same rotational distance
    april_cumulative = np.concatenate([[0], np.cumsum(april_ang_vel)])
    odom_cumulative = np.concatenate([[0], np.cumsum(odom_ang_vel)])

    axes2[1, 1].plot(t, april_cumulative, 'b-', label='AprilTag', alpha=0.7)
    axes2[1, 1].plot(t, odom_cumulative, 'r--', label='Odometry', alpha=0.7)
    axes2[1, 1].set_xlabel('Time (s)')
    axes2[1, 1].set_ylabel('Cumulative Rotation (deg)')
    axes2[1, 1].set_title('Total Rotation Traveled')
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
    t: np.ndarray,
    R_rot_align: np.ndarray | None,
    output_path: Optional[str] = None,
):
    """
    Visualize AprilTag detection and odometry positions projected onto RGB frames.

    Blue circle: AprilTag detected position with odometry orientation arrow
    Red circle: Odometry position with AprilTag orientation arrow
    Green line: Error between them
    """
    # To project odometry back to image, we need the inverse transform
    # april_aligned = R @ april_raw + t  =>  odom frame
    # To go from odom to april_raw: april_raw = R.T @ (odom - t)
    R_inv = R.T

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
            odom_robot = R_inv @ (odom_pos - t)

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
                april_aligned = R @ april_robot + t
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
