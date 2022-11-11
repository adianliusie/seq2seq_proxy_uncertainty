from typing import Tuple
from types import SimpleNamespace

import os
import torch
import logging

from .utils import kl_divergence_loss
from .cross_entropy import CrossEntropyLoss
from models.models import load_model 

# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'DistillationLoss'
]

class DistillationLoss(CrossEntropyLoss):
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)
        self.teacher = self.load_teacher()
        self.temperature = self.args.temperature

    def load_teacher(self):

        # Point to path for model arguments and load
        path = self.args.teacher_path
        argspath = os.path.join(path, "model_args.json")
        teacherargs = self.load_args(argspath)

        # Load teacher model
        teachermodel = load_model(system=teacherargs.transformer)

        # Load teacher parameters
        name = teacherargs.transformer
        teachermodel.load_state_dict(
            torch.load(os.path.join(path, 'models', f'{name}.pt'))
        )

        # Detach teacher parameters
        for param in teachermodel.parameters():
            param.detach_()
        
        # Set in evaluation mode
        teachermodel = teachermodel.eval()

        # Log parameters
        logger.info("Number of parameters in teacher model {:.1f}M".format(
            sum(p.numel() for p in teachermodel.parameters()) / 1e6
        ))
        logger.info("Number of parameters in teacher encoder {:.1f}M".format(
            sum(p.numel() for p in teachermodel.encoder.parameters()) / 1e6
        ))
        logger.info("Number of parameters in teacher decoder {:.1f}M".format(
            sum(p.numel() for p in teachermodel.decoder.parameters()) / 1e6
        ))
        logger.info("Number of parameters in teacher head {:.1f}M".format(
            sum(p.numel() for p in teachermodel.lm_head.parameters()) / 1e6
        ))

        return teachermodel

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

        # Half CE half KL
        loss = 0.50 * (kl + ce)

        # Token level accuracy
        x = (output.logits.argmax(dim = -1) == batch.label_ids)

        # Masked Token level accuracy
        acc = torch.masked_select(x, mask) / mask.sum()
                
        self.record_metrics({
            'loss': loss.item(),
            'ce': ce.item(),
            'kl': kl.item(),
            'acc': acc.sum(),
        }, batch_size = batch.output_numtokens)

        self.record_metrics({
            'sequences': batch.input_ids.size(0),
            'ip-tok': batch.input_numtokens,
            'ip-pad': batch.input_numpadding,
            'op-tok': batch.output_numtokens,
            'op-pad': batch.output_numpadding,
        }, batch_size = 1)

        return loss


