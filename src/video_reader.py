"""
video_reader.py

Goal:
Learn how OpenCV loads videos frame by frame.

A video is simply a sequence of images shown very quickly.
OpenCV allows us to access each image individually.
"""

import cv2


# Open the video file
video = cv2.VideoCapture("videos/sample.mp4")
# cv2.VideoCapture returns the address of an object of 
# type VideoCapture which acts as a video-handler


# Total frame counter
frame_count = 0


while True:

    # Read one frame

    # ret: 
    # True  -> frame successfully read
    # False -> video ended
    # (ret represents return value which is boolean)

    ret, frame = video.read()

    # frame (NumPy Array): The actual image data matrix for that specific frame.

    if not ret:
        break

    frame_count += 1

    print(f"Frame {frame_count}")

    print("Shape:", frame.shape)

    # Show frame

    cv2.imshow("Video Frame", frame)

    # Wait 25 milliseconds

    # If user presses q, exit

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

print(f"\nTotal Frames: {frame_count}")