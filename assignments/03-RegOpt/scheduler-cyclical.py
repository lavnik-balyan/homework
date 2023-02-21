from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler class.

    """

    def __init__(self, optimizer, max_lr=0.001, total_epochs, step_size, mode='triangular', gamma=1.0, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            last_epoch (int): The index of the last epoch. Default: -1.

        """
        # ... Your Code Here ...
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.cycle_steps = 2 * math.floor(total_epochs / step_size)
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
        cycle = math.floor(1 + self.last_epoch / self.step_size)
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            lr = self.max_lr - (self.max_lr / 2) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.max_lr / 2 * max(0, (1 - x))
        elif self.mode == 'exp_range':
            lr = self.max_lr - (self.max_lr / 2) * max(0, (1 - x)) * (self.gamma ** self.last_epoch)
        return [lr * base_lr for base_lr in self.base_lrs]
