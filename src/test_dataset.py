"""
Test our custom dataset.
"""

from pose_dataset import PoseDataset


dataset = PoseDataset(
    "data/pose_landmarks.csv"
)

print("Dataset Size:", len(dataset))

print("\nFirst Sample:\n")

print(dataset[0])

print("\nFeature Count:")

print(dataset[0].shape)