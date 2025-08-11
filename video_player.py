import pickle
import numpy as np
import cv2
import os

# Path to the pickle file
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
pickle_file_path = os.path.join(downloads_dir, "zed_video_frames.pkl")

# Load the NumPy array from the pickle file
with open(pickle_file_path, 'rb') as f:
    video_array = pickle.load(f)

print(f"Loaded video array with shape: {video_array.shape}")

# Replay the video
for frame in video_array:
    # Convert the frame to uint8 type if necessary
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Display the frame
    cv2.imshow("Replayed Video", frame)
    
    # Wait for 33ms to simulate ~30fps video playback
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
