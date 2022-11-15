from types import SimpleNamespace
from typing import Tuple

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
    def eval_forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        # Do standard forward pass
        loss = self.forward(batch)

        # Set number of beams (hardcoded to 12)
        NUM_BEAMS = 4

        # Generate teacher forcing prediction
        force_output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids,
        )

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

        # Get text for teacher forcing predictions
        force_ids = force_output.logits.argmax(dim = -1)
        force_text = self.tokenizer.batch_decode(force_ids, skip_special_tokens=True)

        # Get text of best free running predictions
        free_text = self.tokenizer.batch_decode(free_output.sequences, skip_special_tokens=True)

        for i, j in zip(free_text, batch.label_text):
            print(i)
            print(j)
            print()

        print(len(free_text), len(batch.label_text), len(force_text))
        # Corpus level bleu scoring for teacher-forcing
        bleu_force = sacrebleu.corpus_bleu(
            force_text,
            batch.label_text,
        ).score

        # Corpus level bleu scoring for free-running
        bleu_free = sacrebleu.corpus_bleu(
            free_text,
            batch.label_text,
        ).score

        # Record Bleu scores
        self.record_metrics({
            'bleu_force': bleu_force,
            'bleu_free':  bleu_free,
        }, batch_size = batch.output_numtokens)

        return loss
