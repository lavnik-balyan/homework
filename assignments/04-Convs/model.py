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
        self.conv1 = nn.Conv2d(num_channels, 16, 3, stride=2)
        self.batchnorm = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for CNN.
        """

        x = self.pool(self.batchnorm(F.relu(self.conv1(x))))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
