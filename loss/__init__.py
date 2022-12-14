from .utils import get_entropy
from .utils import kl_divergence_loss
from .cross_entropy import CrossEntropyLoss
from .distillation import DistillationLoss
from .distillation_proxy import DistillationProxyLoss


def load_loss(loss, args, model, tokenizer):
    if loss == 'cross_entropy':
        return CrossEntropyLoss(args, model, tokenizer)
    if loss == 'distillation':
        return DistillationLoss(args, model, tokenizer)
    if loss == 'distillation_proxy':
        return DistillationProxyLoss(args, model, tokenizer)