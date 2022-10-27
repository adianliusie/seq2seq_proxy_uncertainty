from typing import List, Tuple
from types import SimpleNamespace
from collections.abc import Iterator

import torch
import random


class Batcher(object):
    def __init__(self, maxlen: int = 512):
        self.maxlen = maxlen
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def batches(self, data: List, numtokens: int, shuffle: bool = False) ->  Iterator:
        """
        Splits the data into batches and returns them
        """
        examples = self._prepare(data)
    
        batches, step = [], 0
        while step < len(examples):

            # Dynamic batch size based on max tokens
            bsz = numtokens // len(examples[step][1])

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
        input_ids, attention_mask = self._process(input_ids)
        label_ids, _ = self._process(label_ids, pad_id = -100)
        return SimpleNamespace(
            ex_id=ex_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            label_ids=label_ids,
            input_text=input_text,
            label_text=label_text,
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

        # Sort all examples based on length
        examples = sorted(examples, key = lambda x: len(x[1]), reverse=True)

        return examples 

    def _process(self, ids: List, pad_id: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Pads 2D input ids array so that every row has the same length
        """
        # Pad all inputs to this length
        maxlen = max([len(x) for x in ids])

        # All sequences of the same length
        padded_ids = [x + [pad_id] * (maxlen - len(x)) for x in ids]
        mask = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]

        # Create 2D tensors
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    def to(self, device: torch.device):
        self.device = device
    
    def __call__(self, data, numtokens, shuffle=False):
        """
        Routes the main method do the batches function
        """
        return self.batches(data=data, numtokens=numtokens, shuffle=shuffle)
    