import torch
import torch.nn.functional as F

def get_entropy(logits):
    logp = torch.log_softmax(logits, dim = -1)
    entropy = -torch.exp(logp) * logp
    entropy = entropy.sum(-1)
    return entropy

def kl_divergence_loss(input_logits, target_logits, temperature):
    """Computes the temperature annealed kl-divergence
    Gradients only propagated through the input logits."""
    input_lsoftmax = F.log_softmax(input_logits/temperature, dim = -1)
    target_lsoftmax = F.log_softmax(target_logits/temperature, dim = -1)

    loss = torch.exp(target_lsoftmax) * (target_lsoftmax - input_lsoftmax)
    loss = (loss.sum(-1)).mean()
    return loss
