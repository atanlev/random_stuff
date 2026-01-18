"""AprilTag tracker class for detecting and tracking rigid bodies."""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional
from pupil_apriltags import Detector

from .tracker_data_types import RigidBodyConfig, Pose, FrameResult


class AprilTagTracker:
    """
    Tracks rigid bodies using AprilTags in robot coordinate frame.
    Poses are returned in ROS-style coordinates (X=forward, Y=left, Z=up)
    for direct transformation to odometry frame.
    """

    def __init__(
        self,
        rigid_bodies: list[RigidBodyConfig],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_family: str = "tag36h11"
    ):
        self.rigid_bodies = {rb.tag_id: rb for rb in rigid_bodies}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.detector = Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
        )

        self.tag_sizes = {rb.tag_id: rb.tag_size for rb in rigid_bodies}

        self.base_link_tag_id: Optional[int] = None

        for rb in rigid_bodies:
            if rb.name == "base_link":
                self.base_link_tag_id = rb.tag_id
                break

    def _get_tag_pose(self, detection, tag_size: float) -> Pose:
        """Extract pose from a detection using solvePnP, converted to robot frame.

        OpenCV camera frame: X=right, Y=down, Z=forward
        Robot frame (ROS): X=forward, Y=left, Z=up

        Note: The rotation alignment (to match odometry) is computed automatically
        from stationary frames in processing.py, not hardcoded here.
        """
        half_size = tag_size / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ], dtype=np.float64)

        image_points = detection.corners.astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            raise ValueError(f"solvePnP failed for tag {detection.tag_id}")

        R_camera, _ = cv2.Rodrigues(rvec)
        t_camera = tvec.flatten()

        # Transform position from camera frame to robot frame
        # Camera: X=right, Y=down, Z=forward (toward robot)
        # Robot:  X=forward, Y=left, Z=up
        # Camera faces robot, so:
        #   Robot X = -Camera Z
        #   Robot Y = Camera X
        #   Robot Z = -Camera Y
        R_cam_to_robot_pos = np.array([
            [0,  0, -1],
            [1,  0,  0],
            [0, -1,  0],
        ])
        t_robot = R_cam_to_robot_pos @ t_camera

        # solvePnP gives R_camera = rotation of TAG in CAMERA frame
        # We want the robot's orientation, which is the INVERSE (camera in tag frame)
        # Then we need to account for the camera facing the robot
        R_robot = R_camera.T  # Transpose = inverse for rotation matrix

        return Pose(position=t_robot, rotation=R_robot)

    def set_reference_from_frame(self, frame: np.ndarray) -> bool:
        """Check that base_link tag is visible in first frame (for validation)."""
        if self.base_link_tag_id is None:
            raise ValueError("No base_link rigid body configured")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        if self.base_link_tag_id not in detection_by_id:
            print("Error: base_link AprilTag not visible in reference frame")
            return False

        print(f"base_link AprilTag detected in first frame")
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_utc: float) -> FrameResult:
        """Process a single frame and return poses in camera frame for all detected rigid bodies."""
        result = FrameResult(frame_idx=frame_idx, timestamp_utc=timestamp_utc)

        for rb in self.rigid_bodies.values():
            result.poses[rb.name] = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        for tag_id, rb in self.rigid_bodies.items():
            if tag_id in detection_by_id:
                detection = detection_by_id[tag_id]
                pose_camera = self._get_tag_pose(detection, self.tag_sizes[tag_id])
                # Return pose in camera frame (no reference transform)
                result.poses[rb.name] = pose_camera

        return result
