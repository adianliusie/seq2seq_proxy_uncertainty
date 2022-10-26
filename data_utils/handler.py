import random

from tqdm import tqdm
from copy import deepcopy
from typing import Optional, List, Dict
from types import SimpleNamespace
from functools import lru_cache

from .models.tokenizers import load_tokenizer
from .loader import load_nmt_data


__all__ = [
    "DataHandler"
]


def subset(data: Optional[List], lim: Optional[int] = None):
    if data is None: 
        return None

    # Copy data for some reason?
    data = data.copy()

    # Get the same randomiser 
    seed = random.Random(1)
    seed.shuffle(data)
    return data[:lim]


def to_namespace(*args: List):
    def _to_namespace(data: List[Dict]) -> List[SimpleNamespace]:
        return [SimpleNamespace(ex_id = k, **ex) for k, ex in enumerate(data)]

    output = [_to_namespace(split) for split in args]
    return output if len(args) > 1 else output[0]


class DataHandler(object):
    def __init__(self, name: str):
        self.tokenizer = load_tokenizer(name)
    
    @classmethod
    def load_split(cls, dname: str, mode: str, lim: Optional[int] = None):
        split = {'train' : 0, 'dev' : 1, 'test' : 2}
        data = cls.load_data(dname, lim)[split[mode]]
        return data
    
    @staticmethod
    @lru_cache(maxsize = 5)
    def load_data(dname: str, lim: Optional[int] = None):
        train, dev, test = load_nmt_data(dname)
            
        if lim is not None:
            train = subset(train, lim)
            dev   = subset(dev, lim)
            test  = subset(test, lim)
            
        train, dev, test = to_namespace(train, dev, test)
        return train, dev, test
    
    @lru_cache(maxsize = 5)
    def prep_split(self, dname: str, mode: str, lim: Optional[int] = None):
        data = self.load_split(dname, mode, lim)
        output = self._prep_ids(data) 
        return output
    
    @lru_cache(maxsize = 5)
    def prep_data(self, dname, lim: Optional[int] = None):
        train, dev, test = self.load_data(data_name = dname, lim = lim)
        train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        return train, dev, test
        
    def _prep_ids(self, split: List[SimpleNamespace]):
        split = deepcopy(split)
        for ex in tqdm(split):
            input_ids = self.tokenizer(ex.input_text).input_ids
            label_ids = self.tokenizer(ex.label_text).input_ids
            ex.input_ids = input_ids
            ex.label_ids = label_ids
        return split
    