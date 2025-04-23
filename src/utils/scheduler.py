import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import optim


class GPTLrScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, min_lr:float=6e-5, warmup_steps:int=10, decrease_steps:int=50, max_lr: float = None):
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decrease_steps = decrease_steps

        self.current_step = 0

        super().__init__(optimizer)

    @staticmethod
    def _calc_lr(min_lr: float, max_lr: float, warmup: int, decrease: int, step: int):
        if step < warmup:
            lr_out = max_lr * (step + 1) / warmup
        # min lr
        elif step > decrease:
            lr_out = min_lr
        # in between
        else:
            decay_ratio = (step - warmup) / (decrease - warmup)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr_out = min_lr + coeff * (max_lr - min_lr)
        return lr_out

    def get_lr(self):
        return [
            self._calc_lr(self.min_lr, base_lr, self.warmup_steps, self.decrease_steps, self.current_step)
            for base_lr in self.base_lrs
        ]

    def step(self, step=None):
        if step:
            self.current_step = step

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self.current_step += 1
