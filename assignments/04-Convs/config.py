from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip


class CONFIG:
    batch_size = 16
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=2e-3)

    transforms = Compose([ToTensor(), RandomHorizontalFlip()])
