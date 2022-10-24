
import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss

class CrossLoss(BaseLoss):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch:SimpleNamespace)->Tuple[float, dict]:
        model_output = self.model(input_ids=batch.input_ids, 
                                  attention_mask=batch.attention_mask, 
                                  labels=batch.label_ids)
        loss = model_output.loss      
        
        self.record_metrics({
            'loss': loss.item()
        })
        return loss
    
    def generate_text(self, batch, **kwargs):
        # resource for generation: https://huggingface.co/blog/how-to-generate 
        model_preds = self.model.generate(batch.input_ids)

        output = []
        for input_text, label_text, pred in zip(batch.input_text, batch.label_text, model_preds):
            pred_text = self.tokenzier.decode(pred, **kwargs)
            ex_output = {'input_text':input_text, 
                         'pred_text':pred_text,
                         'label_text':label_text}
            output.append(ex_output)
        return output
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)