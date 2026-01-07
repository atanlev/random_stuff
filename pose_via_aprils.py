"""
AprilTag-based rigid body pose tracking from video.

Links AprilTag IDs to named rigid bodies and tracks their 6DoF pose
relative to a reference AprilTag that defines the world coordinate system.
"""
from __future__ import annotations # just because my python is old

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pickle
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

@dataclass
class RigidBodyConfig:
    """Configuration for a rigid body tracked by an AprilTag."""
    name: str
    tag_id: int
    tag_size: float


@dataclass
class Pose:
    """6DoF pose (position + orientation)."""
    position: np.ndarray
    rotation: np.ndarray

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist()
        }

    def inverse(self) -> "Pose":
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.position
        return Pose(position=t_inv, rotation=R_inv)

    def compose(self, other: "Pose") -> "Pose":
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
        self.rigid_bodies = {rb.tag_id: rb for rb in rigid_bodies}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.external_tag_id = external_tag_id
        self.external_tag_size = external_tag_size

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

        self.tag_sizes = {rb.tag_id: rb.tag_size for rb in rigid_bodies}
        if external_tag_id is not None:
            self.tag_sizes[external_tag_id] = external_tag_size

        self.base_link_tag_id: Optional[int] = None
        self.reference_pose: Optional[Pose] = None
        self.calibration_rotation: Optional[np.ndarray] = None

        for rb in rigid_bodies:
            if rb.name == "base_link":
                self.base_link_tag_id = rb.tag_id
                break

    @staticmethod
    def _quaternion_to_rotation_matrix(q) -> np.ndarray:
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

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

    def _apply_calibration(self, pose: Pose) -> Pose:
        if self.calibration_rotation is None:
            return pose
        # R_corrected = R_calib @ R_measured
        return Pose(position=pose.position, 
                    rotation=self.calibration_rotation @ pose.rotation)

    def _transform_to_reference(self, pose: Pose) -> Pose:
        if self.reference_pose is None:
            raise ValueError("Reference pose not set. Call calibrate_from_frame first.")

        R_ref_inv = self.reference_pose.rotation.T
        t_ref = self.reference_pose.position

        pos_in_ref = R_ref_inv @ (pose.position - t_ref)
        rot_in_ref = R_ref_inv @ pose.rotation

        return Pose(position=pos_in_ref, rotation=rot_in_ref)

    def verify_axis_alignment(self, frame: np.ndarray) -> Optional[dict]:
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

        if self.reference_pose is not None:
            external_in_ref = self._transform_to_reference(external_pose)
        else:
            external_in_ref = external_pose

        result = {
            "external_tag_id": self.external_tag_id,
            "position": external_in_ref.position.tolist(),
            "x_axis": external_in_ref.rotation[:, 0].tolist(),
            "y_axis": external_in_ref.rotation[:, 1].tolist(),
            "z_axis": external_in_ref.rotation[:, 2].tolist(),
        }

        print(f"External AprilTag axis verification: position={external_in_ref.position}")
        print(f"  X-axis: {result['x_axis']}")
        print(f"  Y-axis: {result['y_axis']}")
        print(f"  Z-axis: {result['z_axis']}")

        return result

    def calibrate_from_frame(self, frame: np.ndarray, ground_truth_pose=None) -> bool:
        """
        Set reference frame using base_link pose at frame 0 and compute calibration.
        Calculates a rotation matrix that aligns the AprilTag measurement with ground truth.
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

        if ground_truth_pose is not None:
            R_ground_truth = self._quaternion_to_rotation_matrix(ground_truth_pose.orientation)
            
            # Since R_measured in reference frame at T=0 is Identity:
            # R_calib = R_ground_truth @ I^T = R_ground_truth
            self.calibration_rotation = R_ground_truth

            print("Calibration rotation computed from ground truth")

        self.verify_axis_alignment(frame)
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float) -> FrameResult:
        """Process a single frame and return poses for all detected rigid bodies."""
        result = FrameResult(frame_idx=frame_idx, timestamp_sec=timestamp_sec)

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
                pose_calibrated = self._apply_calibration(pose_ref)
                
                result.poses[rb.name] = pose_calibrated

        return result


# =============================================================================
# Tracker Configuration
# =============================================================================

TAG_SIZE_M = 0.116

RIGID_BODIES = [
    RigidBodyConfig(name="base_link", tag_id=75, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="left_foot_ee", tag_id=18, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="right_foot_ee", tag_id=109, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="left_hip", tag_id=91, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="right_hip", tag_id=97, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="torso", tag_id=5, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="left_arm", tag_id=87, tag_size=TAG_SIZE_M),
    RigidBodyConfig(name="right_arm", tag_id=84, tag_size=TAG_SIZE_M),
]

EXTERNAL_TAG_ID = None
EXTERNAL_TAG_SIZE = TAG_SIZE_M

# ZED camera intrinsics
CAMERA_MATRIX = np.array([
    [520.5241088867188, 0, 649.637939453125],
    [0, 520.5241088867188, 368.625732421875],
    [0, 0, 1]
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


def create_tracker() -> AprilTagTracker:
    return AprilTagTracker(
        rigid_bodies=RIGID_BODIES,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        external_tag_id=EXTERNAL_TAG_ID,
        external_tag_size=EXTERNAL_TAG_SIZE,
    )


def process_video(
    video_path: str,
    tracker: AprilTagTracker,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> list[FrameResult]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    print(f"Processing video: {video_path} (FPS: {fps}, Frames: {total_frames})")

    # Read first frame for calibration
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {start_frame}")

    if not tracker.calibrate_from_frame(frame):
        raise ValueError("Failed to calibrate - base_link tag not visible in first frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    results = []
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process_frame(frame, frame_idx, frame_idx / fps)
        results.append(result)

        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx}/{end_frame}")

    cap.release()
    print(f"Processed {len(results)} frames")
    return results


def project_pose_to_image(pose: Pose, camera_matrix: np.ndarray, reference_pose: Pose) -> tuple[int, int]:
    """Project a 3D pose back to 2D image coordinates."""
    pos_camera = reference_pose.rotation @ pose.position + reference_pose.position
    x = camera_matrix[0, 0] * pos_camera[0] / pos_camera[2] + camera_matrix[0, 2]
    y = camera_matrix[1, 1] * pos_camera[1] / pos_camera[2] + camera_matrix[1, 2]
    return int(x), int(y)


def visualize_on_video(
    video_path: str,
    results: list[FrameResult],
    tracker: AprilTagTracker,
    output_path: Optional[str] = None,
    start_frame: int = 0,
):
    colors_bgr = {
        'base_link': (0, 0, 255),
        'left_foot_ee': (255, 0, 0),
        'right_foot_ee': (0, 255, 0),
        'left_hip': (0, 165, 255),
        'right_hip': (255, 0, 255),
        'torso': (255, 255, 0),
        'left_arm': (255, 0, 255),
        'right_arm': (0, 255, 255),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print("Visualizing - press 'q' to quit, 'space' to pause")

    paused = False
    for result in results:
        ret, frame = cap.read()
        if not ret:
            break

        for name, pose in result.poses.items():
            if pose is None:
                continue

            color = colors_bgr.get(name, (255, 255, 255))
            x, y = project_pose_to_image(pose, tracker.camera_matrix, tracker.reference_pose)

            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)
            cv2.putText(frame, name, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw coordinate axes
            axis_length = 0.05
            axes_3d = np.array([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
            axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

            for axis, ax_color in zip(axes_3d, axis_colors):
                axis_end_ref = Pose(position=pose.position + pose.rotation @ axis, rotation=pose.rotation)
                ax_x, ax_y = project_pose_to_image(axis_end_ref, tracker.camera_matrix, tracker.reference_pose)
                cv2.line(frame, (x, y), (ax_x, ax_y), ax_color, 2)

        cv2.putText(frame, f"Frame: {result.frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if writer: writer.write(frame)
        cv2.imshow('AprilTag Tracking', frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"Saved output video to: {output_path}")
    cv2.destroyAllWindows()


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def results_to_7dof_dict(results: list[FrameResult]) -> dict[str, np.ndarray]:
    """Convert tracking results to a dict of 7DoF arrays (pos + quat)."""
    body_names = set().union(*(r.poses.keys() for r in results))
    n_frames = len(results)
    data = {name: np.full((7, n_frames), np.nan) for name in body_names}

    for frame_idx, result in enumerate(results):
        for name in body_names:
            pose = result.poses.get(name)
            if pose is not None:
                data[name][:3, frame_idx] = pose.position
                data[name][3:, frame_idx] = rotation_matrix_to_quat(pose.rotation)
    return data


def save_results_pickle(results: list[FrameResult], output_path: str):
    data = results_to_7dof_dict(results)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved 7DoF results to: {output_path}")
    for name, arr in data.items():
        print(f"  {name}: shape={arr.shape}, valid_frames={np.sum(~np.isnan(arr[0, :]))}")


if __name__ == "__main__":
    video_path = "/home/ethanl/Downloads/zed_video_output_april.mp4"

    tracker = create_tracker()
    results = process_video(video_path, tracker)

    pickle_path = video_path.replace('.mp4', '_7dof.pkl')
    save_results_pickle(results, pickle_path)

    visualize_on_video(video_path, results, tracker)

    # Plot 3D positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = {
        'base_link': 'red', 'left_foot_ee': 'blue', 'right_foot_ee': 'green',
        'left_hip': 'orange', 'right_hip': 'purple', 'torso': 'cyan',
        'left_arm': 'magenta', 'right_arm': 'yellow'
    }

    positions = {name: [] for name in colors.keys()}
    for result in results:
        for name, pose in result.poses.items():
            if pose:
                positions[name].append(pose.position)

    for name, pos_list in positions.items():
        if pos_list:
            pos_array = np.array(pos_list)
            ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
                    c=colors.get(name, 'black'), label=name, alpha=0.7)
            ax.scatter(*pos_array[0], c=colors.get(name, 'black'), s=100, marker='o')
            ax.scatter(*pos_array[-1], c=colors.get(name, 'black'), s=100, marker='x')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('AprilTag Trajectories')
    ax.legend()
    plt.show()