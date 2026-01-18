"""Main entry point for the AprilTag tracker package."""
import argparse

from .tracker_io import load_zed_frames, load_walk_log
from .config import create_tracker, AUTO_TIME_OFFSET, ZED_FRAMES_PATH, WALK_LOG_PATH, ODOM_FRAME
from .processing import process_frames, compare_positions, find_best_offset
from .visualization import plot_comparison, visualize_on_frames


def main():
    parser = argparse.ArgumentParser(description="AprilTag tracking comparison with odometry")
    parser.add_argument("--no-time-offset", action="store_true",
                        help="Disable automatic time offset detection")
    parser.add_argument("--time-offset-ms", type=float, default=None,
                        help="Manual time offset in milliseconds (overrides auto-detection)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (just show stats)")
    args = parser.parse_args()

    zed_frames_path = ZED_FRAMES_PATH
    walk_log_path = WALK_LOG_PATH

    print("Loading data...")
    zed_frames, intrinsics = load_zed_frames(zed_frames_path)
    walk_log = load_walk_log(walk_log_path, frame_filter=ODOM_FRAME)

    if ODOM_FRAME is not None:
        print(f"Filtered odometry to frame: {ODOM_FRAME}")

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

    tracker = create_tracker(intrinsics)
    april_results, matched_odom, frames_used, R, t, R_rot_align, time_offset = process_frames(
        zed_frames, walk_log, tracker, time_offset_s=time_offset_s
    )
    print(f"\nTime offset applied: {time_offset * 1000:.1f} ms")

    # Search for best tag-to-baselink offset
    print("\nSearching for optimal tag-to-baselink offset...")
    best_offset = find_best_offset(april_results, matched_odom, R, t, R_rot_align)

    comparison = compare_positions(april_results, matched_odom, R, t, R_rot_align,
                                   tag_offset=best_offset)

    if not args.no_viz:
        plot_comparison(comparison)

        # Visualize on RGB frames (using only frames that overlap with odometry)
        visualize_on_frames(
            frames_used, april_results, matched_odom, tracker,
            R, t, R_rot_align, output_path="april_vs_odom.mp4"
        )


if __name__ == "__main__":
    main()
