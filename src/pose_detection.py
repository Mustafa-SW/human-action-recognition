"""
pose_detection.py

Goal:
Detect human body landmarks using MediaPipe.

MediaPipe converts a person into
33 body keypoints.

Each keypoint contains:
- x coordinate
- y coordinate
- z coordinate
- visibility score
"""

import cv2
import mediapipe as mp

# -----------------------------
# Initialize MediaPipe Pose
# -----------------------------

mp_pose = mp.solutions.pose

# Utility for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Create pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Load Video
# -----------------------------

video = cv2.VideoCapture("videos/sample.mp4")

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Resize 4K frame
    frame = cv2.resize(frame, (640, 360))

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    results = pose.process(rgb)

    # Draw skeleton if detected
    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()