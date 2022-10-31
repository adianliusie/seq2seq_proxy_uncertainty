from typing import List, Tuple
from types import SimpleNamespace
from collections.abc import Iterator

import torch
import random


class Batcher(object):
    def __init__(self, maxlen: int = 512):
        self.maxlen = maxlen
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def batches(self, data: List, numtokens: int, numsequences: int, shuffle: bool = False) ->  Iterator:
        """
        Splits the data into batches and returns them
        """
        examples = self._prepare(data)
    
        batches, step = [], 0
        while step < len(examples):

            # Dynamic batch size based on max tokens in target sequence
            bsz = min(numtokens // len(examples[step][2]), numsequences)

            # Stop early
            if step + bsz > len(examples):
                break

            # Store batch
            batches.append(examples[step:step + bsz])

            # Update step
            step += bsz

        if shuffle: 
            random.shuffle(batches)

        for batch in batches:
            yield self.batchify(batch)

    def batchify(self, batch: List[List]) -> SimpleNamespace:
        """
        Each input is input ids and mask for utt, + label
        """
        ex_id, input_ids, label_ids, input_text, label_text = zip(*batch)  
        input_ids, attention_mask, ip_numtokens, ip_numpadding = self._process(input_ids)
        label_ids, _, op_numtokens, op_numpadding = self._process(label_ids, pad_id = -100)
        return SimpleNamespace(
            ex_id=ex_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            label_ids=label_ids,
            input_text=input_text,
            label_text=label_text,
            input_numtokens=ip_numtokens,
            input_numpadding=ip_numpadding,
            output_numtokens=op_numtokens,
            output_numpadding=op_numpadding,
        )
    
    def _prepare(self, data: List) -> List[List]:
        """
        Sequence classification input data preparation
        """
        examples = []
        for ex in data:
            ex_id = ex.ex_id
            input_ids = ex.input_ids
            label_ids = ex.label_ids
            input_text = ex.input_ids
            label_text = ex.label_text

            # Skip all examples larger than limit
            if len(input_ids) > self.maxlen: 
                continue

            examples.append([ex_id, input_ids, label_ids, input_text, label_text])

        # Sort all examples based on average length
        examples = sorted(examples, key = lambda x: len(x[1]) + len(x[2]), reverse=True)

        return examples 

    def _process(self, ids: List, pad_id: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor, int, int]:
        """
        Pads 2D input ids array so that every row has the same length
        """
        # Pad all inputs to this length
        maxlen = max([len(x) for x in ids])

        # Get the number of tokens
        numtokens = sum(len(x) for x in ids)
        numpadding = maxlen * len(ids) - numtokens

        # All sequences of the same length
        padded_ids = [x + [pad_id] * (maxlen - len(x)) for x in ids]
        mask = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]

        # Create 2D tensors
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask, numtokens, numpadding

    def to(self, device: torch.device):
        self.device = device
    
    def __call__(self, data, numtokens, numsequences, shuffle=False):
        """
        Routes the main method do the batches function
        """
        return self.batches(data=data, numtokens=numtokens, numsequences=numsequences, shuffle=shuffle)
    