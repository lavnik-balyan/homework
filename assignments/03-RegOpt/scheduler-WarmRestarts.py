from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler class.

    """

    def __init__(self, optimizer, t_0=500, t_mult=1, lr_max=0.001, lr_min=0.0001, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.

        """
        # ... Your Code Here ...
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cycle_len = self.t_0
        self.iteration = 0
        self.restart_iteration = 0
        self.cycle_count = 0
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
        self.iteration += 1
        if self.iteration > self.cycle_len:
            self.iteration = 1
            self.restart_iteration = 1
            self.cycle_len = int(self.t_0 * (self.t_mult ** self.cycle_count))
            self.cycle_count += 1

        x = math.pi * self.restart_iteration / self.cycle_len
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(x))
        self.restart_iteration += 1
        temp = [lr * base_lr for base_lr in self.base_lrs]
        print(temp)
        return temp
