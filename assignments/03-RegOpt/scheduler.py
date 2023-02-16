from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler class.

    """

    def __init__(self, optimizer, step_size=30, gamma=0.1, total_iters=5, power=1.0, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
            last_epoch (int): The index of the last epoch. Default: -1.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.gamma = gamma
        self.total_iters = total_iters
        self.power = power
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
        # return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]
