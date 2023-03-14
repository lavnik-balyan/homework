from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
<<<<<<< HEAD
    batch_size = 40
=======
    batch_size = 32
>>>>>>> 9227f6f2593e34bbd50349edf01ca5771940c3fb
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=3e-3)

    transforms = Compose([ToTensor()])
