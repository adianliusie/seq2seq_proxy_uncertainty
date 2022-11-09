from typing import Tuple
from types import SimpleNamespace

import os
import torch
import logging

import scipy
import scipy.stats
import pytorch_wrapper as pw

import torchsort
from torchsort import soft_rank, soft_sort

from .utils import kl_divergence_loss
from .utils import get_sentence_confidence_unc, get_sentence_entropy_unc
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


def encoder_mse(encoder_preds, teacher_encoder_preds, mask):
    loss = (encoder_preds - teacher_encoder_preds) ** 2
    loss = pw.functional.masked_mean_pooling(loss, mask, dim = 1)
    loss = loss.mean()
    return loss


class DistillationProxyLoss(DistillationLoss):
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)

        # Set loss arguments 
        self.distillation_w = args.proxy_distillation_weight_start
        self.encoder_w = args.proxy_encoder_weight
        self.set_scheduling(args)

        # Set model arguments
        self.model.set_arguments(args)
        
        # Set the scoring function
        self.proxy_gen = get_sentence_confidence_unc
        if args.proxy_entropy:
            self.proxy_gen = get_sentence_entropy_unc

    def rank_loss(self, output, teacher_output, label_mask):

        target_scalars = self.proxy_gen(
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

        # Encoder loss
        enc = encoder_mse(
            encoder_preds = output.encoder_last_hidden_state, 
            teacher_encoder_preds = teacher_output.encoder_last_hidden_state,
            mask = batch.attention_mask,
        )

        # Ranking loss
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
            'enc': enc.item(),
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

        return rl + self.distillation_w * kl + self.encoder_w * enc

    def set_scheduling(self, args):
        w_start, w_end = args.proxy_distillation_weight_start, args.proxy_distillation_weight_end
        if args.proxy_distillation_weight_end is not None:
            self.step_size = (w_end - w_start)/args.num_steps
        else:
            self.step_size = 0

    def step(self):
        self.distillation_w += self.step_size

    
