from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomRotation,
    RandomHorizontalFlip,
)


class CONFIG:
    batch_size = 64
    num_epochs = 6
    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
            RandomRotation(10),
            RandomHorizontalFlip(),
        ]
    )
