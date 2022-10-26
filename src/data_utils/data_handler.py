import random

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from ..models.tokenizers import load_tokenizer
from .load_nmt_data      import load_nmt_data

#== Main DataLoader class =========================================================================#
class DataHandler:
    def __init__(self, trans_name:str):
        self.tokenizer = load_tokenizer(trans_name)
    
    #== Data loading utils ========================================================================#
    @classmethod
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data
    
    @staticmethod
    @lru_cache(maxsize=5)
    def load_data(data_name:str, lim=None):
        train, dev, test = load_nmt_data(data_name)
            
        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)
            
        train, dev, test = to_namespace(train, dev, test)
        return train, dev, test
    
    #== Data processing (i.e. tokenizing text) ====================================================#
    @lru_cache(maxsize=5)
    def prep_split(self, data_name:str, mode:str, lim=None):
        data = self.load_split(data_name, mode, lim)
        output = self._prep_ids(data) 
        return output
    
    @lru_cache(maxsize=5)
    def prep_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        return train, dev, test
        
    def _prep_ids(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            input_ids = self.tokenizer(ex.input_text).input_ids
            label_ids = self.tokenizer(ex.label_text).input_ids
            ex.input_ids = input_ids
            ex.label_ids = label_ids
        return split_data
    
#== Misc utils functions ==========================================================================#
def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]

def to_namespace(*args:List):
    def _to_namespace(data:List[dict])->List[SimpleNamespace]:
        return [SimpleNamespace(ex_id=k, **ex) for k, ex in enumerate(data)]

    output = [_to_namespace(split) for split in args]
    return output if len(args)>1 else output[0]
