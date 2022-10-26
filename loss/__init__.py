from .utils import get_entropy
from .utils import kl_divergence_loss
from .cross_entropy import CrossEntropyLoss


def load_model_loss(loss, args, model):
    if loss == 'cross_entropy':
        return CrossEntropyLoss(args, model)