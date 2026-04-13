"""LeNet model definition for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """Classic LeNet-5 style CNN adapted for 28x28 MNIST images.

    Original LeNet-5 was designed for 32x32 inputs. Here we keep the same
    conv/pool pattern, and adapt the first fully-connected layer input size
    to match MNIST 28x28 after two conv+pool stages.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Input: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # After conv1: (N, 6, 24, 24)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # After pool1: (N, 6, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # After conv2: (N, 16, 8, 8)
        # After pool2: (N, 16, 4, 4)

        # For 28x28 MNIST, flattened feature size is 16*4*4=256.
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
