"""
Train a simple pose model.
"""

import torch

from torch.utils.data import DataLoader

from labeled_pose_dataset import LabeledPoseDataset

from model import PoseClassifier


dataset = LabeledPoseDataset(
    "data/pose_landmarks.csv"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


model = PoseClassifier()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 5


for epoch in range(epochs):

    total_loss = 0

    for features, labels in loader:

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} | Loss = {total_loss:.4f}"
    )

    torch.save(
    model.state_dict(),
    "pose_model.pth"
)

print("\nModel saved")