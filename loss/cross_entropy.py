from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss


__all__ = [
    'CrossEntropyLoss'
]

class CrossEntropyLoss(BaseLoss):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:

        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        loss = output.loss      
        
        self.record_metrics({
            'loss': loss.item(),
            'ce': loss.item(),
        }, batch_size = batch.output_numtokens)

        self.record_metrics({
            'num-sequences': batch.input_ids.size(0),
            'ip-num-tokens': batch.input_numtokens,
            'ip-num-padding': batch.input_numpadding,
            'op-num-tokens': batch.output_numtokens,
            'op-num-padding': batch.output_numpadding,
        }, batch_size = 1)

        return loss
