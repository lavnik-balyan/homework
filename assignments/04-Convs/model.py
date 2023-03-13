import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A CNN for getting 0.55 accuracy on the CIFAR-10 dataset as quickly as possible.

    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Create a new model class.

        Arguments:
            num_channels (int): Number of input channels.
            num_classes (int): Number of output classes.

        """

        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for CNN.

        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
