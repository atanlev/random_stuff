import pyzed.sl as sl
import cv2
import os
import pickle
import time
from copy import copy

# Set configuration parameters
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30  # Set the frame rate

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Set runtime parameters
runtime_parameters = sl.RuntimeParameters()

# Prepare a single image container
image = sl.Mat()

# List to store frames with timestamps
frames_data = []

# Output file path
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
pickle_file_path = os.path.join(downloads_dir, "zed_frames.pkl")

print("Recording frames, press 'q' to stop...")

while True:

    # Grab a new frame
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image
        zed.retrieve_image(image, sl.VIEW.LEFT)

        # Get UTC timestamp
        timestamp = time.time()

        # Convert to NumPy array
        frame = copy(image.get_data())

        # Store frame with timestamp
        frames_data.append({
            'timestamp': timestamp,
            'frame': frame[..., :3]  # Only store 3 channels (RGB)
        })

        # Display the frame
        cv2.imshow("ZED Camera", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera
zed.close()
cv2.destroyAllWindows()

# Save frames with timestamps to pickle file
print(f"Saving {len(frames_data)} frames to pickle file...")
with open(pickle_file_path, 'wb') as f:
    pickle.dump(frames_data, f)

print(f"Captured {len(frames_data)} frames and saved to {pickle_file_path}")
