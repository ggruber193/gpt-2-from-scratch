from typing import Any

from torch import Tensor
from torchmetrics import Metric
import torch
from torch.nn import functional as F


class HellaSwagAccuracy(Metric):
    def __init__(self, n_answers=4, **kwargs):
        super().__init__(**kwargs)

        self._n_answers = n_answers
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, tokens: Tensor, mask: Tensor, target: Tensor):
        if preds.min() < 0 or preds.max() > 1:
            shift_logits = preds[..., :-1, :].contiguous()
            shift_tokens = tokens[..., 1:].contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            shift_mask = mask[..., 1:].contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            preds = avg_loss.view(-1, self._n_answers).argmin(dim=1)

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
