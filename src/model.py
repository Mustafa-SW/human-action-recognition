"""
Simple neural network
for pose classification.
"""

import torch.nn as nn


class PoseClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(132, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)

        )

    def forward(self, x):

        return self.network(x)