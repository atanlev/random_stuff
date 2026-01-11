"""Data types for AprilTag tracking."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class FrameResult:
    """Tracking results for a single frame."""
    frame_idx: int
    timestamp_utc: float
    poses: dict[str, Optional[Pose]] = field(default_factory=dict)
