from typing import Tuple
from types import SimpleNamespace

import os
import torch

from utils.meter import AverageMeter
from utils.general import load_json


class BaseLoss(object):
    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key].reset()

    def record_metrics(self, batch_metrics, batch_size = 1):
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value, batch_size)

    def load_args(self, path: str) -> SimpleNamespace:
        args = load_json(path)
        return SimpleNamespace(**args)

    def forward(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def eval_forward(self, batch):
        return self.forward(batch)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def step(self):
        pass