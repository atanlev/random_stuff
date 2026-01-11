"""AprilTag tracker class for detecting and tracking rigid bodies."""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional
from pupil_apriltags import Detector

from .tracker_data_types import RigidBodyConfig, Pose, FrameResult


class AprilTagTracker:
    """
    Tracks rigid bodies using AprilTags with base_link at frame 0 as reference.
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
        self.reference_pose: Optional[Pose] = None

        for rb in rigid_bodies:
            if rb.name == "base_link":
                self.base_link_tag_id = rb.tag_id
                break

    def _get_tag_pose(self, detection, tag_size: float) -> Pose:
        """Extract pose from a detection using solvePnP."""
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

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        return Pose(position=t, rotation=R)

    def _transform_to_reference(self, pose: Pose) -> Pose:
        if self.reference_pose is None:
            raise ValueError("Reference pose not set.")

        R_ref_inv = self.reference_pose.rotation.T
        t_ref = self.reference_pose.position

        pos_in_ref = R_ref_inv @ (pose.position - t_ref)
        rot_in_ref = R_ref_inv @ pose.rotation

        return Pose(position=pos_in_ref, rotation=rot_in_ref)

    def set_reference_from_frame(self, frame: np.ndarray) -> bool:
        """Set reference frame using base_link pose from first frame."""
        if self.base_link_tag_id is None:
            raise ValueError("No base_link rigid body configured")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        if self.base_link_tag_id not in detection_by_id:
            print("Error: base_link AprilTag not visible in reference frame")
            return False

        base_link_detection = detection_by_id[self.base_link_tag_id]
        self.reference_pose = self._get_tag_pose(
            base_link_detection,
            self.tag_sizes[self.base_link_tag_id]
        )

        print(f"Reference set to base_link at frame 0: pos={self.reference_pose.position}")
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_utc: float) -> FrameResult:
        """Process a single frame and return poses for all detected rigid bodies."""
        result = FrameResult(frame_idx=frame_idx, timestamp_utc=timestamp_utc)

        for rb in self.rigid_bodies.values():
            result.poses[rb.name] = None

        if self.reference_pose is None:
            print("Warning: Reference pose not set, returning empty result")
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        for tag_id, rb in self.rigid_bodies.items():
            if tag_id in detection_by_id:
                detection = detection_by_id[tag_id]
                pose_camera = self._get_tag_pose(detection, self.tag_sizes[tag_id])
                pose_ref = self._transform_to_reference(pose_camera)
                result.poses[rb.name] = pose_ref

        return result
