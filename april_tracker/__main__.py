"""Main entry point for the AprilTag tracker package."""
from .tracker_io import load_zed_frames, load_walk_log
from .config import create_tracker
from .processing import process_frames, compare_positions
from .visualization import plot_comparison, visualize_on_frames


def main():
    zed_frames_path = "zed_frames.pkl"
    walk_log_path = "walk_log.pkl"

    print("Loading data...")
    zed_frames = load_zed_frames(zed_frames_path)
    walk_log = load_walk_log(walk_log_path)

    print(f"Loaded {len(zed_frames)} frames, {len(walk_log)} odometry entries")
    print(f"Frame timestamps: {zed_frames[0]['timestamp']:.3f} - {zed_frames[-1]['timestamp']:.3f}")
    print(f"Odom timestamps:  {walk_log[0]['timestamp_utc']:.3f} - {walk_log[-1]['timestamp_utc']:.3f}")

    tracker = create_tracker()
    april_results, matched_odom, frames_used, R, scale, t, R_rot_align = process_frames(
        zed_frames, walk_log, tracker
    )

    comparison = compare_positions(april_results, matched_odom, R, scale, t, R_rot_align)

    plot_comparison(comparison)

    # Visualize on RGB frames (using only frames that overlap with odometry)
    visualize_on_frames(
        frames_used, april_results, matched_odom, tracker,
        R, scale, t, R_rot_align, output_path="april_vs_odom.mp4"
    )


if __name__ == "__main__":
    main()
