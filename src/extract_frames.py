"""
extract_frames.py

Goal:
Convert a video into individual images.

Instead of saving every frame, we save every 10th frame
to reduce storage requirements.

We also resize frames from 4K resolution to 640x360,
which is much more manageable for machine learning tasks.
"""

import cv2
import os


# Open the video file
video = cv2.VideoCapture("videos/sample.mp4")

# Create output directory if it does not exist
os.makedirs("frames", exist_ok=True)

# Tracks current frame number in video
frame_id = 0

# Tracks how many frames we actually save
saved_count = 0

# Save every Nth frame
FRAME_SKIP = 10

# Desired output size
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360


while True:

    # Read one frame from video
    ret, frame = video.read()

    # Stop when video ends
    if not ret:
        break

    # Save every 10th frame
    if frame_id % FRAME_SKIP == 0:

        # Resize frame
        frame = cv2.resize(
            frame,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        )

        # Create filename
        filename = f"frames/frame_{saved_count}.jpg"

        # Save image
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        saved_count += 1

    frame_id += 1


# Release video resources
video.release()

print("\n===== Extraction Complete =====")
print(f"Total Frames in Video : {frame_id}")
print(f"Frames Saved          : {saved_count}")
print(f"Sampling Rate         : Every {FRAME_SKIP}th frame")
print(f"Output Resolution     : {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")