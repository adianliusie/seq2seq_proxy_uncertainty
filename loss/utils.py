import torch

__all__ = [
    "get_entropy",
    "kl_divergence_loss",
]


def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy based on final dimension.
    """
    lsoftmax = torch.log_softmax(logits, dim = -1)
    entropy = -torch.exp(lsoftmax) * lsoftmax
    entropy = entropy.sum(-1)
    return entropy


def kl_divergence_loss(input_logits: torch.Tensor, target_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Computes the temperature annealed kl-divergence
    Gradients should only propagated through the input logits.
    """
    input_lsoftmax = torch.log_softmax(input_logits/temperature, dim = -1)
    target_lsoftmax = torch.log_softmax(target_logits/temperature, dim = -1)

    loss = torch.exp(target_lsoftmax) * (target_lsoftmax - input_lsoftmax)
    loss = (loss.sum(-1)).mean()
    return loss
