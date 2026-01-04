import cv2
import numpy as np


def detect_red_objects(image_path, output_path=None):
    """
    Detect red objects in an image using HSV color space.

    Args:
        image_path: Path to input image
        output_path: Optional path to save visualization

    Returns:
        mask: Binary mask where red objects are white (255)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV, so we need two ranges
    # Lower red range (0-10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # Upper red range (160-180)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes

    if output_path:
        # Create visualization with red objects highlighted
        result = img.copy()
        # Highlight detected red regions in green for visibility
        result[mask > 0] = (result[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Draw contours around detected objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)

        cv2.imwrite(output_path, result)

    return mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python red_detection.py <image_path> [output_path]")
        sys.exit(1)

    img_path = '/home/ethanl/Downloads/test_can_image.png'
    out_path = "red_detected.jpg"

    mask = detect_red_objects(img_path, out_path)
    print(f"Red detection saved to {out_path}. Red pixels: {np.sum(mask > 0)}")
