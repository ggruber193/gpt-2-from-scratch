from pathlib import Path

import torch
from typing import Literal, Any
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from dataclasses import dataclass
from accelerate.tracking import GeneralTracker
from torch.fx.experimental.optimization import optimize_for_inference
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.model.gpt import GPT2


def autodetect_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device

class ExperimentSetup:
    def __init__(self, project_name: str, project_dir: str | Path, log_dir: str | Path = None,
                 trackers: list[GeneralTracker] = None):
        self._project_name = project_name
        self._project_dir = project_dir
        self._log_dir = log_dir
        self._trackers = trackers

    @property
    def project_configuration(self):
        return ProjectConfiguration(self._project_dir, self._log_dir)
    @property
    def project_name(self):
        return self._project_name
    @property
    def trackers(self):
        return self._trackers


class Trainer:
    def __init__(self, device="auto", grad_accum_steps: int = 1, mixed_precision: bool = False,
                 epochs: int = 1, steps: int = None, val_interval=1.0, log_interval: int = 10, gradient_clipping: float = 1.0):
        self.device = autodetect_device()
        self._max_epochs = epochs
        self._max_steps = steps
        self._val_interval = val_interval
        self._log_steps = log_interval
        self._gradient_clipping = gradient_clipping
        self._grad_accum_steps = grad_accum_steps
        self._current_step = 0
        self._model: GPT2 = None
        self._optimizer: Optimizer = None
        self._scheduler: LRScheduler = None
        self._autocast_dtype = torch.bfloat16 if mixed_precision else torch.float32

    def _reset(self):
        self._current_step = 0
        self._model: GPT2 = None
        self._optimizer: Optimizer = None
        self._scheduler: LRScheduler = None

    def _train_step(self, batch):
        x, y = batch
        logits, loss = self._model(x, y)
        return loss

    def _val_step(self, batch):
        x, y = batch
        logits, loss = self._model(x)
        self._accelerator.log({"val/loss": loss.item()}, step=self._current_step)
        return loss

    def _validation(self, dataloader: DataLoader):
        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                self._val_step(batch)

    def fit(self, model: GPT2, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader = None, scheduler: LRScheduler = None):
        self._reset()
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._model = model

        self._model.to(self.device)

        steps_per_epoch = len(train_loader)
        val_steps_interval = int(self._val_interval * steps_per_epoch)
        for iteration in range(self._max_epochs):
            self._model.train()
            for step, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                batch = (x, y)
                loss_accum = 0
                for ministep in range(self._grad_accum_steps):
                    with torch.autocast(device_type=self.device, dtype=self._autocast_dtype):
                        loss = self._train_step(batch)
                    loss = loss / self._grad_accum_steps
                    loss_accum += loss.detach()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clipping)
                self._optimizer.step()
                if self._scheduler is not None:
                    self._scheduler.step()
                    self._optimizer.zero_grad()

                if self._current_step % self._log_steps == 0:
                    print({"train/loss": loss_accum.item(), "step": self._current_step})

                if self._current_step % val_steps_interval:
                    if val_loader is not None:
                        self._validation(val_loader)
                self._current_step += 1
