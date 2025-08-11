import pyzed.sl as sl
import numpy as np
import cv2
import pickle
import os
from copy import copy

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30  # Set the frame rate
init_params.depth_mode = sl.DEPTH_MODE.NEURAL

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Set runtime parameters
runtime_parameters = sl.RuntimeParameters()
# Prepare containers for color and depth images
color_image = sl.Mat()
depth_image = sl.Mat()

# Lists to hold color and depth frames
color_frames = []
depth_frames = []

print("Recording video, press 'q' to stop...")

while True:
    # Grab a new frame
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the color image
        zed.retrieve_image(color_image, sl.VIEW.LEFT)
        color_frame = copy(color_image.get_data())

        # Retrieve the depth image
        zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
        depth_frame = copy(depth_image.get_data())

        # Ensure the color frame has 3 channels (RGB)
        if color_frame.shape[2] == 4:  # RGBA
            color_frame = color_frame[:, :, :3]  # Remove the alpha channel

        # Ensure the depth frame is a single channel
        if depth_frame.ndim == 2:
            depth_frame_rgb = cv2.cvtColor(depth_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            depth_frame_rgb = cv2.cvtColor(depth_frame[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Append to frames list
        color_frames.append(color_frame)
        depth_frames.append(depth_frame)

        # Ensure both frames are the same size before combining
        if color_frame.shape[:2] == depth_frame_rgb.shape[:2]:
            # Display the color image and depth image side by side
            combined_frame = np.hstack((color_frame, depth_frame_rgb))
            cv2.imshow("ZED Camera - Color and Depth", combined_frame)
        else:
            print("Frame size mismatch")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Convert lists of frames to NumPy arrays
color_video_array = np.array(color_frames)
depth_video_array = np.array(depth_frames)

# Release the camera
zed.close()
cv2.destroyAllWindows()

# Path to save the pickle file
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
pickle_file_path = os.path.join(downloads_dir, "zed_video_frames.pkl")

# Save the NumPy arrays to a pickle file
try:
    with open(pickle_file_path, 'wb') as f:
        pickle.dump({'color': color_video_array, 'depth': depth_video_array}, f)
    print(f"Captured {len(color_frames)} color frames and {len(depth_frames)} depth frames")
except Exception as e:
    print(f"Error saving pickle file: {e}")
