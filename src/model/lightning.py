import sys
import os
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent))

import time

import torch
from lightning_fabric.utilities import measure_flops
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from dataclasses import dataclass
from torch.nn import functional as F

import torchmetrics

from gpt import GPT2, GPT2Config, GPT2ConfigPretrained
from src.utils.scheduler import GPTLrScheduler
from metrics import HellaSwagAccuracy

@dataclass
class GPT2TrainingConfig:
    lr: float = 6e-4
    weight_decay: float = 0.1
    b0: float = 0.9
    b1: float = 0.95
    min_lr: float = lr * 0.1
    warmup_steps: int = 100
    decay_steps: int = 2000

class GPTLightning(LightningModule):
    def __init__(self,
                 model_config: GPT2Config,
                 training_config: GPT2TrainingConfig = GPT2TrainingConfig(),
                 model=None):
        super().__init__()
        hyperparams = model_config.__dict__ | training_config.__dict__
        self.save_hyperparameters(hyperparams)
        if model is not None:
            self.model = model
        else:
            self.model = GPT2(model_config)

        self.train_metrics = torchmetrics.MetricCollection({
            "perplexity": torchmetrics.text.Perplexity()
        }, prefix="train/")
        self.perplexity = torchmetrics.text.Perplexity()
        self.hellaswag_acc = HellaSwagAccuracy()

    def _common_step(self, batch, batch_idx, stage="train"):
        if len(batch) == 3:
            tokens, masks, labels = batch
            logits, y_hat = self.hellaswag_step(batch)
            loss = F.cross_entropy(y_hat.view(-1, 4), labels)
            self.hellaswag_acc(logits, tokens, masks, labels)
            self.log(f"{stage}/hellaswag_accuracy", self.hellaswag_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            x, y = batch
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            self.perplexity(logits, y)
            self.log("train/perplexity", self.perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logits, loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        logits, loss = self._common_step(batch, batch_idx, "valid")
        return loss


    def training_step(self, batch, batch_idx):
        t0 = time.time()
        x, *_ = batch
        logits, loss = self._common_step(batch, batch_idx, "train")
        t1 = time.time()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("tok/sec", x.shape[0]*x.shape[1]/(t1-t0), prog_bar=True, logger=False, on_step=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        print(self.hparams)
        optimizer = self.model.configure_optimizers(weight_decay=self.hparams.weight_decay, learning_rate=self.hparams.lr)
        scheduler = GPTLrScheduler(optimizer, min_lr=self.hparams.min_lr, warmup_steps=self.hparams.warmup_steps,
                                   decrease_steps=self.hparams.decay_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2", training_config: GPT2TrainingConfig = GPT2TrainingConfig()):
        return cls(GPT2ConfigPretrained(model_type), training_config)

    @classmethod
    def from_model(cls, model: GPT2, training_config: GPT2TrainingConfig = GPT2TrainingConfig()):
        return cls(model.config, training_config, model)

    def hellaswag_step(self, batch):
        tokens, mask, label = batch
        logits = self.model(tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        preds = sum_loss.view(-1, 4).argmin(dim=1)
        preds_norm = avg_loss.view(-1, 4).argmin(dim=1)
        return logits, avg_loss


if __name__ == '__main__':
    model = GPTLightning.from_pretrained("gpt2")
    print(model.hparams)
