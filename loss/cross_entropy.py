from types import SimpleNamespace
from typing import Tuple

import torch

from .base import BaseLoss


__all__ = [
    'CrossEntropyLoss'
]

class CrossEntropyLoss(BaseLoss):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:

        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        # Cross entropy loss
        loss = output.loss  

        # Masking out all non-labels
        mask = batch.label_ids != -100

        # Token level accuracy
        x = (output.logits.argmax(dim = -1) == batch.label_ids)

        # Masked Token level accuracy
        acc = torch.masked_select(x, mask) 
                
        self.record_metrics({
            'loss': loss.item(),
            'ce': loss.item(),
            'acc': acc.sum() / mask.sum(),
        }, batch_size = batch.output_numtokens)

        self.record_metrics({
            'sequences': batch.input_ids.size(0),
            'ip-tok': batch.input_numtokens,
            'ip-pad': batch.input_numpadding,
            'op-tok': batch.output_numtokens,
            'op-pad': batch.output_numpadding,
        }, batch_size = 1)

        return loss


