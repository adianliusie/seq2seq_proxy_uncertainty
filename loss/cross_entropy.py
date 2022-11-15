from types import SimpleNamespace
from typing import Tuple
from collections.abc import Iterator

import torch
import sacrebleu

from .base import BaseLoss


__all__ = [
    'CrossEntropyLoss'
]

class CrossEntropyLoss(BaseLoss):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:

        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        # Cross entropy loss
        loss = output.loss  

        # Masking out all non-labels
        mask = batch.label_ids != -100

        # Token level accuracy
        x = (output.logits.argmax(dim = -1) == batch.label_ids)

        # Masked Token level accuracy
        acc = torch.masked_select(x, mask) 
                
        self.record_metrics({
            'loss': loss.item(),
            'ce': loss.item(),
            'acc': acc.sum() / mask.sum(),
        }, batch_size = batch.output_numtokens)

        self.record_metrics({
            'sequences': batch.input_ids.size(0),
            'ip-tok': batch.input_numtokens,
            'ip-pad': batch.input_numpadding,
            'op-tok': batch.output_numtokens,
            'op-pad': batch.output_numpadding,
        }, batch_size = 1)

        return loss

    @torch.no_grad()
    def eval_forward(self, loader: Iterator) -> Tuple[float, dict]:
        
        # Metrics to store
        pairs = {'ref': [], 'prd': []}

        # Hardcoded number of beams
        NUM_BEAMS = 4

        for batch in loader:

            # Generate teacher forcing prediction
            output = self.model(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                labels = batch.label_ids,
            )

            # Cross entropy loss
            loss = output.loss  

            # Masking out all non-labels
            mask = batch.label_ids != -100

            # Token level accuracy
            x = (output.logits.argmax(dim = -1) == batch.label_ids)

            # Masked Token level accuracy
            acc = torch.masked_select(x, mask) 

            # Generate free running prediction
            free_output = self.model.generate(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                max_length = 256,
                num_beams = NUM_BEAMS,
                length_penalty = 0.6,
                no_repeat_ngram_size = 5,
                num_return_sequences = 1,
                output_scores = True,
                return_dict_in_generate = True,
            )

            # Get the beam and decode
            beams = free_output.sequences
            texts = self.tokenizer.batch_decode(beams, skip_special_tokens=True)

            # Store results for corpus level computation
            pairs['ref'].extend(batch.label_text)
            pairs['prd'].extend(texts)

            # Record accuracy scores
            self.record_metrics({
                'acc': acc.sum() / mask.sum(),
            }, batch_size = mask.sum())

        # Corpus level scoring free-running
        freescore = sacrebleu.corpus_bleu(
            pairs['prd'],
            [pairs['ref']],
        ).score

        # Record accuracy scores
        self.record_metrics({
            'loss': -freescore,
            'bleu': freescore,
        }, batch_size = 1)

        return loss
