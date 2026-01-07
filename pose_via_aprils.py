"""
AprilTag-based rigid body pose tracking from video.

Links AprilTag IDs to named rigid bodies and tracks their 6DoF pose
relative to a reference AprilTag that defines the world coordinate system.
"""
from __future__ import annotations

# =============================================================================
# AprilTag Pose Tracking Classes
# =============================================================================
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pupil_apriltags import Detector
import pyzed.sl as sl

@dataclass
class RigidBodyConfig:
    """Configuration for a rigid body tracked by an AprilTag."""
    name: str
    tag_id: int
    tag_size: float  # Tag size in meters


@dataclass
class Pose:
    """6DoF pose (position + orientation)."""
    position: np.ndarray  # [x, y, z] in meters
    rotation: np.ndarray  # 3x3 rotation matrix

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist()
        }

    def inverse(self) -> "Pose":
        """Return the inverse of this pose transform."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.position
        return Pose(position=t_inv, rotation=R_inv)

    def compose(self, other: "Pose") -> "Pose":
        """Compose this pose with another: self @ other."""
        R_new = self.rotation @ other.rotation
        t_new = self.rotation @ other.position + self.position
        return Pose(position=t_new, rotation=R_new)


@dataclass
class FrameResult:
    """Tracking results for a single frame."""
    frame_idx: int
    timestamp_sec: float
    poses: dict[str, Optional[Pose]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "poses": {
                name: pose.to_dict() if pose else None
                for name, pose in self.poses.items()
            }
        }


class AprilTagTracker:
    """
    Tracks rigid bodies using AprilTags with base_link at frame 0 as reference.

    The base_link pose at frame 0 defines the world coordinate system origin.
    An external AprilTag is used only to verify axis alignment between robot and camera.
    """

    def __init__(
        self,
        rigid_bodies: list[RigidBodyConfig],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        external_tag_id: Optional[int] = None,
        external_tag_size: float = 0.05,
        tag_family: str = "tag36h11"
    ):
        """
        Args:
            rigid_bodies: List of rigid body configurations to track
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            external_tag_id: Optional external AprilTag ID for axis verification
            external_tag_size: Size of external tag in meters
            tag_family: AprilTag family (default: tag36h11)
        """
        self.rigid_bodies = {rb.tag_id: rb for rb in rigid_bodies}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.external_tag_id = external_tag_id
        self.external_tag_size = external_tag_size

        # Extract camera params for detector
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        self.camera_params = (fx, fy, cx, cy)

        self.detector = Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
        )

        # Build tag_id -> size mapping
        self.tag_sizes = {}
        for rb in rigid_bodies:
            self.tag_sizes[rb.tag_id] = rb.tag_size
        if external_tag_id is not None:
            self.tag_sizes[external_tag_id] = external_tag_size

        # Reference frame: base_link pose at frame 0 (camera frame)
        self.base_link_tag_id: Optional[int] = None
        self.reference_pose: Optional[Pose] = None  # base_link pose in camera frame at frame 0

        # Calibration rotation: corrects AprilTag orientation to match ground truth
        # R_calibrated = R_calib @ R_measured
        self.calibration_rotation: Optional[np.ndarray] = None

        # Find base_link tag_id
        for rb in rigid_bodies:
            if rb.name == "base_link":
                self.base_link_tag_id = rb.tag_id
                break

    @staticmethod
    def _quaternion_to_rotation_matrix(q) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w

        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R

    def _get_tag_pose(self, detection, tag_size: float) -> Pose:
        """Extract pose from a detection with known tag size using solvePnP."""
        # Define 3D object points for the tag corners (centered at origin)
        half_size = tag_size / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],  # top-left
            [ half_size,  half_size, 0],  # top-right
            [ half_size, -half_size, 0],  # bottom-right
            [-half_size, -half_size, 0],  # bottom-left
        ], dtype=np.float64)

        # Get detected corner points (2D image coordinates)
        image_points = detection.corners.astype(np.float64)

        # Solve PnP to get rotation and translation
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            raise ValueError(f"solvePnP failed for tag {detection.tag_id}")

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        return Pose(position=t, rotation=R)

    def _apply_calibration(self, pose: Pose) -> Pose:
        """Apply calibration rotation to correct AprilTag orientation error."""
        if self.calibration_rotation is None:
            return pose

        # Apply calibration: R_corrected = R_calib @ R_measured
        corrected_rotation = self.calibration_rotation @ pose.rotation
        return Pose(position=pose.position, rotation=corrected_rotation)

    def _transform_to_reference(self, pose: Pose) -> Pose:
        """Transform a pose from camera frame to reference (base_link frame 0) frame."""
        if self.reference_pose is None:
            raise ValueError("Reference pose not set. Call calibrate_from_frame first.")

        # Inverse of reference pose
        R_ref_inv = self.reference_pose.rotation.T
        t_ref = self.reference_pose.position

        # Transform position
        pos_in_ref = R_ref_inv @ (pose.position - t_ref)

        # Transform rotation
        rot_in_ref = R_ref_inv @ pose.rotation

        return Pose(position=pos_in_ref, rotation=rot_in_ref)

    def verify_axis_alignment(self, frame: np.ndarray) -> Optional[dict]:
        """
        Verify axis alignment using external AprilTag.

        Returns axis comparison info if external tag is visible, None otherwise.
        """
        if self.external_tag_id is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        if self.external_tag_id not in detection_by_id:
            print("Warning: External AprilTag not visible for axis verification")
            return None

        external_detection = detection_by_id[self.external_tag_id]
        external_pose = self._get_tag_pose(external_detection, self.external_tag_size)

        # Transform to reference frame
        if self.reference_pose is not None:
            external_in_ref = self._transform_to_reference(external_pose)
        else:
            external_in_ref = external_pose

        # Extract axis directions for comparison
        x_axis = external_in_ref.rotation[:, 0]
        y_axis = external_in_ref.rotation[:, 1]
        z_axis = external_in_ref.rotation[:, 2]

        result = {
            "external_tag_id": self.external_tag_id,
            "position": external_in_ref.position.tolist(),
            "x_axis": x_axis.tolist(),
            "y_axis": y_axis.tolist(),
            "z_axis": z_axis.tolist(),
        }

        print(f"External AprilTag axis verification: position={external_in_ref.position}")
        print(f"  X-axis: {x_axis}")
        print(f"  Y-axis: {y_axis}")
        print(f"  Z-axis: {z_axis}")

        return result

    def calibrate_from_frame(self, frame: np.ndarray, ground_truth_pose=None) -> bool:
        """
        Set reference frame using base_link pose at frame 0 and compute calibration.

        At frame 0, the robot's internal current_location is used as ground truth.
        We compute a calibration rotation that aligns the AprilTag measurement
        to match the ground truth orientation.

        Args:
            frame: BGR image (frame 0)
            ground_truth_pose: Robot SDK Pose from current_location (ground truth)

        Returns:
            True if calibration successful, False otherwise
        """
        if self.base_link_tag_id is None:
            raise ValueError("No base_link rigid body configured")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {d.tag_id: d for d in detections}

        if self.base_link_tag_id not in detection_by_id:
            print("Error: base_link AprilTag not visible in calibration frame")
            return False

        base_link_detection = detection_by_id[self.base_link_tag_id]
        self.reference_pose = self._get_tag_pose(
            base_link_detection,
            self.tag_sizes[self.base_link_tag_id]
        )

        print(f"Reference set to base_link at frame 0: pos={self.reference_pose.position}")

        # Compute calibration rotation if ground truth is provided
        if ground_truth_pose is not None:
            # Get ground truth rotation from quaternion
            R_ground_truth = self._quaternion_to_rotation_matrix(ground_truth_pose.orientation)

            # Get measured rotation (in reference frame, which at frame 0 is identity)
            # At frame 0, the measured rotation in reference frame is identity by definition
            # But we need to correct future measurements, so we compute:
            # R_calib = R_ground_truth @ R_measured^T
            # This way: R_ground_truth = R_calib @ R_measured

            # The AprilTag gives us R_measured in camera frame
            # After transform_to_reference at frame 0, R_measured_ref = I (identity)
            # But the actual AprilTag rotation in camera frame is self.reference_pose.rotation
            # We want to find R_calib such that when we transform any future pose:
            # The corrected rotation aligns with ground truth

            # At frame 0: R_apriltag_ref = I (by definition of reference)
            # But ground truth says it should be R_ground_truth
            # So R_calib = R_ground_truth @ I^T = R_ground_truth
            self.calibration_rotation = R_ground_truth

            print("Calibration rotation computed from ground truth")
            print(f"  Ground truth quaternion: x={ground_truth_pose.orientation.x:.4f}, "
                       f"y={ground_truth_pose.orientation.y:.4f}, "
                       f"z={ground_truth_pose.orientation.z:.4f}, "
                       f"w={ground_truth_pose.orientation.w:.4f}")

        # Verify axis alignment with external tag if configured
        self.verify_axis_alignment(frame)

        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float) -> FrameResult:
        """
        Process a single frame and return poses for all detected rigid bodies.

        Args:
            frame: BGR image from video
            frame_idx: Frame index
            timestamp_sec: Timestamp in seconds

        Returns:
            FrameResult with poses relative to base_link at frame 0
        """
        result = FrameResult(frame_idx=frame_idx, timestamp_sec=timestamp_sec)

        # Initialize all rigid bodies as not detected
        for rb in self.rigid_bodies.values():
            result.poses[rb.name] = None

        if self.reference_pose is None:
            print("Warning: Reference pose not set, returning empty result")
            return result

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all tags
        detections = self.detector.detect(gray)

        # Build detection lookup
        detection_by_id = {d.tag_id: d for d in detections}

        # Process each rigid body
        for tag_id, rb in self.rigid_bodies.items():
            if tag_id in detection_by_id:
                detection = detection_by_id[tag_id]
                tag_size = self.tag_sizes[tag_id]

                # Get pose in camera frame
                pose_camera = self._get_tag_pose(detection, tag_size)

                # Transform to reference frame (base_link at frame 0)
                pose_ref = self._transform_to_reference(pose_camera)

                # Apply calibration to correct orientation error
                pose_calibrated = self._apply_calibration(pose_ref)
                result.poses[rb.name] = pose_calibrated

        return result


# =============================================================================
# Tracker Configuration
# =============================================================================

# Tag size in meters (from URDF: 0.116m = 11.6cm)
TAG_SIZE_M = 0.116

# Rigid bodies to track (tag_id should be updated to actual values)
RIGID_BODIES = [
    RigidBodyConfig(name="base_link", tag_id=91, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="left_foot_ee", tag_id=109, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="right_foot_ee", tag_id=18, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="left_hip", tag_id=97, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="right_hip", tag_id=84, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="torso", tag_id=5, tag_size=TAG_SIZE_M),
]

# External AprilTag for axis verification (set to None to disable)
EXTERNAL_TAG_ID = 91
EXTERNAL_TAG_SIZE = TAG_SIZE_M


def init_zed_camera() -> sl.Camera:
    """
    Initialize and open ZED camera.

    Returns:
        Opened ZED camera instance
    """
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720                                                                                                                                                                                                                                                             
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    print("ZED camera initialized successfully")
    return zed


def get_zed_camera_intrinsics(zed: sl.Camera) -> tuple[np.ndarray, np.ndarray]:
    """
    Get camera intrinsics from an open ZED camera.

    Args:
        zed: Open ZED camera instance

    Returns:
        Tuple of (camera_matrix, dist_coeffs)
    """
    # Get calibration parameters for left camera
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    left_cam = calibration_params.left_cam

    # Build camera matrix
    fx = left_cam.fx
    fy = left_cam.fy
    cx = left_cam.cx
    cy = left_cam.cy

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Get distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array(left_cam.disto, dtype=np.float64)

    print(f"ZED camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"ZED distortion coeffs: {dist_coeffs}")

    return camera_matrix, dist_coeffs


def create_tracker(zed: sl.Camera) -> AprilTagTracker:
    """Create and return an AprilTag tracker instance with ZED camera intrinsics."""
    camera_matrix, dist_coeffs = get_zed_camera_intrinsics(zed)

    return AprilTagTracker(
        rigid_bodies=RIGID_BODIES,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        external_tag_id=EXTERNAL_TAG_ID,
        external_tag_size=EXTERNAL_TAG_SIZE,
    )


if __name__ == "__main__":
    # Initialize ZED camera and tracker
    zed = init_zed_camera()
    tracker = create_tracker(zed)

    # Grab first frame and calibrate
    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3].copy()  # BGR

        # For calibration, you would get ground_truth_pose from your robot SDK:
        # ground_truth_pose = robot.get_current_location()
        # if tracker.calibrate_from_frame(frame, ground_truth_pose=ground_truth_pose):
        #     print("AprilTag tracker calibrated successfully with ground truth alignment")

        # Calibrate without ground truth (just sets reference frame)
        if tracker.calibrate_from_frame(frame):
            print("AprilTag tracker calibrated (reference frame set)")
        else:
            print("Failed to calibrate AprilTag tracker - base_link tag not visible")

    # Example: process frames in a loop
    # frame_idx = 0
    # while zed.grab() == sl.ERROR_CODE.SUCCESS:
    #     zed.retrieve_image(image, sl.VIEW.LEFT)
    #     frame = image.get_data()[:, :, :3].copy()
    #     result = tracker.process_frame(frame, frame_idx, frame_idx / 30.0)
    #     for name, pose in result.poses.items():
    #         if pose:
    #             print(f"Frame {frame_idx} - {name}: pos={pose.position}")
    #     frame_idx += 1

    zed.close()
