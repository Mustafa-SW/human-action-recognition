"""
Learn batching using DataLoader.
"""

from pose_dataset import PoseDataset

from torch.utils.data import DataLoader


dataset = PoseDataset(
    "data/pose_landmarks.csv"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


for batch in loader:

    print("Batch Shape:")

    print(batch.shape)

    break