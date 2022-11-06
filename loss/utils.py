import torch
import pytorch_wrapper as pw

__all__ = [
    "get_entropy",
    "get_sentence_entropy_unc",
    "get_confidence",
    "get_sentence_confidence_unc",
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


def get_sentence_entropy_unc(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the sentence entropy.
    """
    entropy = get_entropy(logits)
    entropy = pw.functional.masked_mean_pooling(entropy, mask, dim = 1)
    return entropy


def get_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy based on final dimension.
    """
    lsoftmax = torch.log_softmax(logits, dim = -1)
    confidence = lsoftmax.max(dim = -1).values
    return confidence


def get_sentence_confidence_unc(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the sentence confidence.
    """
    confidence = get_confidence(logits)
    confidence = pw.functional.masked_mean_pooling(confidence, mask, dim = 1)
    return -confidence


def kl_divergence_loss(input_logits: torch.Tensor, target_logits: torch.Tensor, temperature: float, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the temperature annealed kl-divergence
    Gradients should only propagated through the input logits.
    """
    input_lsoftmax = torch.log_softmax(input_logits/temperature, dim = -1)
    target_lsoftmax = torch.log_softmax(target_logits/temperature, dim = -1)

    loss = torch.exp(target_lsoftmax) * (target_lsoftmax - input_lsoftmax)
    loss = temperature ** 2 * loss.sum(-1)

    # Masked select
    loss = torch.masked_select(loss, mask)
    loss = loss.sum() / mask.sum()

    return loss
