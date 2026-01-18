"""I/O functions for loading data files."""
from __future__ import annotations

import pickle
from typing import Optional


def load_zed_frames(pkl_path: str) -> list[dict]:
    """Load zed_frames.pkl - list of {'timestamp': float, 'frame': np.ndarray}"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_walk_log(pkl_path: str, frame_filter: Optional[str] = None) -> list[dict]:
    """Load walk_log.pkl - list of odometry entries.

    Args:
        pkl_path: Path to pickle file
        frame_filter: If specified, filter entries where entry['frame'] == frame_filter.
                     If None, no filtering is applied (assumes direct x, y, z, qx, qy, qz, qw format)

    Returns:
        List of odometry dictionaries with 'timestamp_utc', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # If frame_filter is specified, filter entries
    if frame_filter is not None:
        # Check if entries have 'frame' attribute
        if data and 'frame' in data[0]:
            data = [entry for entry in data if entry.get('frame') == frame_filter]
        else:
            print(f"Warning: ODOM_FRAME={frame_filter} specified but odometry entries don't have 'frame' attribute")

    return data


def find_closest_odom(timestamp_utc: float, walk_log: list[dict], max_time_diff: float = 1/15) -> Optional[dict]:
    """Find the walk_log entry with the closest timestamp to the given timestamp.

    Returns None if the closest entry is more than max_time_diff seconds away.
    """
    if not walk_log:
        return None

    best_idx = 0
    best_diff = abs(walk_log[0]['timestamp_utc'] - timestamp_utc)

    for i, entry in enumerate(walk_log):
        diff = abs(entry['timestamp_utc'] - timestamp_utc)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    if best_diff > max_time_diff:
        return None

    return walk_log[best_idx]
