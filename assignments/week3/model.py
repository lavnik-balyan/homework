import torch
import torch.nn as nn
from typing import Callable

class MLP(torch.nn.module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = activation
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
