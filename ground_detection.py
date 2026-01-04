import cv2
import numpy as np
from ultralytics import SAM


def load_depth(image_path, depth_base_dir="/home/ethanl/images_from_walk/depth"):
    """Load depth map corresponding to an RGB image."""
    basename = image_path.split("/")[-1].replace(".png", ".bin")
    depth_path = f"{depth_base_dir}/{basename}"
    depth = np.fromfile(depth_path, dtype=np.float32).reshape((1200, 1920))
    return depth


def detect_ground(image_path, depth_path, output_path=None, point_prompts=None, max_depth_m=4):
    """
    Detect ground/floor using MobileSAM segmentation.

    Args:
        image_path: Path to input image
        output_path: Optional path to save visualization
        point_prompts: Optional list of (x, y) points on the floor.
                      If None, uses bottom-center region as default prompts.
        max_depth_m: Maximum depth in meters to keep (default 2.0m)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    # Load MobileSAM model
    model = SAM("mobile_sam.pt")

    # Generate floor point prompts if not provided
    # Default: sample points in the bottom portion of the image (likely floor)
    if point_prompts is None:
        point_prompts = [
            [w // 2, int(h * 0.85)],      # Bottom center
            [w // 4, int(h * 0.9)],       # Bottom left
            [3 * w // 4, int(h * 0.9)],   # Bottom right
            [w // 2, int(h * 0.75)],      # Mid-bottom center
        ]

    # All points are positive (on the floor)
    labels = [1] * len(point_prompts)

    # Run MobileSAM prediction
    results = model.predict(
        image_path,
        points=[point_prompts],
        labels=[labels],
        verbose=False
    )

    # Extract the mask from results
    if results and len(results) > 0 and results[0].masks is not None:
        # Get the mask data and convert to numpy
        mask_data = results[0].masks.data.cpu().numpy()
        # Combine all masks if multiple, take the first one for floor
        mask = (mask_data[0] * 255).astype(np.uint8)
        # Resize mask to original image size if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        print("Warning: No mask generated, returning empty mask")
        mask = np.zeros((h, w), dtype=np.uint8)

    # Filter mask by depth - keep only pixels within max_depth_m
    depth = load_depth(depth_path)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_mask = (depth > 0) & (depth <= max_depth_m)
    mask = mask * depth_mask.astype(np.uint8)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Label 0 is background, find largest among labels 1+
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest_label) * 255).astype(np.uint8)

    if output_path:
        # Create visualization
        result = img.copy()
        result[mask > 0] = (result[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Draw the prompt points
        for pt in point_prompts:
            cv2.circle(result, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

        cv2.imwrite(output_path, result)

    return mask

if __name__ == "__main__":
    img_path = "/home/ethanl/images_from_walk/frame_00000.png"
    depth_path = "/home/ethanl/images_from_walk/depth/frame_00000.bin"
    out_path ="ground_detected.jpg"
    diraction_vector = [0, 1, 0]  # Example direction vector

    mask = detect_ground(img_path, depth_path, out_path)
    print(f"Ground mask saved. Floor pixels: {np.sum(mask > 0)}")
