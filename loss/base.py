import torch

from utils.meter import AverageMeter


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

    def forward(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def eval_forward(self, batch):
        return self.forward(batch)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)