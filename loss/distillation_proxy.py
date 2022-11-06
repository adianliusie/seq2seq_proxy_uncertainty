from typing import Tuple
from types import SimpleNamespace

import os
import torch
import logging

import scipy
import scipy.stats

import torchsort
from torchsort import soft_rank, soft_sort

from .utils import kl_divergence_loss
from .utils import get_sentence_confidence, get_sentence_entropy
from .distillation import DistillationLoss

# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'DistillationProxyLoss'
]


def smooth_rank_loss(input_scalars, target_scalars, param):
    """
    Computes a spearman rank approximated loss.
    Gradients only propagated through the input scalars.
    """

    # Compute the soft rank correlation score
    rank1 = soft_rank(input_scalars.unsqueeze(0), regularization_strength=param)
    rank2 = soft_rank(target_scalars.unsqueeze(0), regularization_strength=param)

    # Normalize and compute batch spearman
    rank1 = (rank1 - rank1.mean())/rank1.norm()
    rank2 = (rank2 - rank2.mean())/rank2.norm()

    spearman_loss = -(rank1 * rank2).sum()
    return spearman_loss


class DistillationProxyLoss(DistillationLoss):
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)

    def rank_loss(self, output, teacher_output, label_mask):

        target_scalars = get_sentence_confidence(
            teacher_output.logits, 
            mask = label_mask
        )

        loss = smooth_rank_loss(
            input_scalars = output.proxies, 
            target_scalars = target_scalars, 
            param = self.args.proxy_regularization_strength
        )
        
        # For spearman tracking
        target_scalars = target_scalars.clone().detach().cpu()
        input_scalars = output.proxies.clone().detach().cpu()

        # Compute correlations
        spear = scipy.stats.spearmanr(input_scalars, target_scalars)[0]
        return loss, spear


    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:

        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                labels = batch.label_ids
            )

        # Cross entropy loss
        ce = output.loss  

        # Masking out all non-labels
        mask = batch.label_ids != -100

        # Distillation loss
        kl = kl_divergence_loss(
            input_logits=output.logits,
            target_logits=teacher_output.logits,
            temperature=self.temperature,
            mask=mask
        )

        rl, spear = self.rank_loss(
            output = output,
            teacher_output = teacher_output,
            label_mask = mask,
        )

        # Token level accuracy
        x = (output.logits.argmax(dim = -1) == batch.label_ids)

        # Masked Token level accuracy
        acc = torch.masked_select(x, mask) / mask.sum()
                
        self.record_metrics({
            'loss': -spear,
            'ce': ce.item(),
            'kl': kl.item(),
            'acc': acc.sum(),
        }, batch_size = batch.output_numtokens)

        self.record_metrics({
            'rl': rl.item(),
            'spear': spear,
            'sequences': batch.input_ids.size(0),
            'ip-tok': batch.input_numtokens,
            'ip-pad': batch.input_numpadding,
            'op-tok': batch.output_numtokens,
            'op-pad': batch.output_numpadding,
        }, batch_size = 1)

        return kl + rl


