from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts


class CustomLRScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler class.

    """

    def __init__(self, optimizer, T_0=32, T_mult=16, eta_min=0.0001, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations for the first restart.
            T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            last_epoch (int, optional): The index of last epoch. Default: -1.

        """
        # ... Your Code Here ...
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate list.

        Returns:
            None

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
