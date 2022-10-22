import torch
from types import SimpleNamespace
import torch.nn.functional as F

class BaseCrossLoss:
    def __init__(self, model):
        self.model = model

    def forward(self, batch):
        output = self.model(input_ids=batch.input_ids, 
                            attention_mask=batch.attention_mask, 
                            labels=batch.label_ids)
        loss = output.loss      
        return SimpleNamespace(loss=loss)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)