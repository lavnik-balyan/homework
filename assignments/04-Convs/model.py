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
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(num_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
=======
        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)
        # self.fc2 = nn.Linear(128, num_classes)
>>>>>>> a2f31d8e4507e25ffdffb7660ab3ce9c445d7dce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for CNN.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))

<<<<<<< HEAD
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
=======
        x = x.view(-1, 32 * 8 * 8)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
>>>>>>> a2f31d8e4507e25ffdffb7660ab3ce9c445d7dce
        return x
