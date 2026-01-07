import pyzed.sl as sl
import cv2
import os
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

# Get the dimensions of the frame
zed.retrieve_image(image, sl.VIEW.LEFT)
frame_shape = image.get_data().shape
frame_height, frame_width = frame_shape[:2]

# Define the codec and create a VideoWriter object
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
video_file_path = os.path.join(downloads_dir, "zed_video_output_april.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(video_file_path, fourcc, 30.0, (frame_width, frame_height))

print("Recording video, press 'q' to stop...")

while True:

    # Grab a new frame
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        
        # Convert to NumPy array
        frame = copy(image.get_data())

        # Write the frame to the video file
        out.write(frame[..., :3])  # Ensure only 3 channels are written

        # Display the frame
        cv2.imshow("ZED Camera", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and video writer
zed.close()
out.release()
cv2.destroyAllWindows()

print(f"Captured video and saved to {video_file_path}")
