"""Main entry point for the AprilTag tracker package."""
import argparse

from .tracker_io import load_zed_frames, load_walk_log
from .config import create_tracker, AUTO_TIME_OFFSET, ZED_FRAMES_PATH, WALK_LOG_PATH
from .processing import process_frames, compare_positions
from .visualization import plot_comparison, visualize_on_frames


def main():
    parser = argparse.ArgumentParser(description="AprilTag tracking comparison with odometry")
    parser.add_argument("--no-time-offset", action="store_true",
                        help="Disable automatic time offset detection")
    parser.add_argument("--time-offset-ms", type=float, default=None,
                        help="Manual time offset in milliseconds (overrides auto-detection)")
    args = parser.parse_args()

    zed_frames_path = ZED_FRAMES_PATH
    walk_log_path = WALK_LOG_PATH

    print("Loading data...")
    zed_frames = load_zed_frames(zed_frames_path)
    walk_log = load_walk_log(walk_log_path)

    print(f"Loaded {len(zed_frames)} frames, {len(walk_log)} odometry entries")
    print(f"Frame timestamps: {zed_frames[0]['timestamp']:.3f} - {zed_frames[-1]['timestamp']:.3f}")
    print(f"Odom timestamps:  {walk_log[0]['timestamp_utc']:.3f} - {walk_log[-1]['timestamp_utc']:.3f}")

    # Determine time offset
    if args.no_time_offset or not AUTO_TIME_OFFSET:
        time_offset_s = 0.0
    elif args.time_offset_ms is not None:
        time_offset_s = args.time_offset_ms / 1000.0
    else:
        time_offset_s = None  # Auto-detect

    tracker = create_tracker()
    april_results, matched_odom, frames_used, R, scale, t, R_rot_align, time_offset = process_frames(
        zed_frames, walk_log, tracker, time_offset_s=time_offset_s
    )
    print(f"\nTime offset applied: {time_offset * 1000:.1f} ms")

    comparison = compare_positions(april_results, matched_odom, R, scale, t, R_rot_align)

    plot_comparison(comparison)

    # Visualize on RGB frames (using only frames that overlap with odometry)
    visualize_on_frames(
        frames_used, april_results, matched_odom, tracker,
        R, scale, t, R_rot_align, output_path="april_vs_odom.mp4"
    )


if __name__ == "__main__":
    main()
