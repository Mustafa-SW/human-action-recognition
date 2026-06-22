"""
extract_landmarks.py

Goal:
Extract MediaPipe pose landmarks
and save them to a CSV file.

Each frame becomes one row.
"""

import cv2
import mediapipe as mp
import pandas as pd
import os


# -----------------------------
# MediaPipe Setup
# -----------------------------

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Video Setup
# -----------------------------

video = cv2.VideoCapture("videos/sample.mp4")

rows = []

frame_id = 0


while True:

    ret, frame = video.read()

    if not ret:
        break

    # Save every 10th frame
    if frame_id % 10 != 0:
        frame_id += 1
        continue

    frame = cv2.resize(frame, (640, 360))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        row = [frame_id]

        for landmark in results.pose_landmarks.landmark:

            row.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        rows.append(row)

        print(f"Processed frame {frame_id}")

    frame_id += 1


video.release()

columns = ["frame_id"]

for i in range(33):

    columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}",
        f"visibility_{i}"
    ])

df = pd.DataFrame(rows, columns=columns)

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/pose_landmarks.csv",
    index=False
)

print("\nSaved dataset")
print(df.head())