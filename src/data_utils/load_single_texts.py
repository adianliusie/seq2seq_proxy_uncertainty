import random
import re

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset
from functools import lru_cache

#== Main loading function =================================================================================# 
class SingleText(TypedDict):
    """Output example formatting (only here for documentation)"""
    text : str
    label : int

SINGLE_TASKS = ['imdb', 'rt', 'sst', 'yelp', 'boolq']
def load_single_texts(data_name)->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    if 'biased' in data_name:  train, dev, test = create_len_bias(data_name)
    elif data_name == 'imdb':  train, dev, test = load_imdb()
    elif data_name == 'rt':    train, dev, test = load_rotten_tomatoes()
    elif data_name == 'sst':   train, dev, test = load_sst()
    elif data_name == 'yelp':  train, dev, test = load_yelp()
    elif data_name == 'boolq': train, dev, test = load_boolq()
    else: raise ValueError(f"invalid single text dataset name: {data_name}")
    return train, dev, test
    
#== Individual Data set Loader Functions =================================================================#
def load_imdb()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("imdb")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    train, dev, test = _remove_html_tags(train, dev, test)
    return train, dev, test

def load_yelp()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("yelp_review_full")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    return train, dev, test

def load_boolq()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("super_glue", "boolq")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test   = list(dataset['validation'])
    train, dev, test = _rename_keys(train, dev, test, old_key='passage', new_key='text')
    return train, dev, test

def load_rotten_tomatoes()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    return train, dev, test

def load_sst()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset('glue', 'sst2')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence', new_key='text')
    return train, dev, test

#== Helper Methods for processing the data sets ========================================================#
def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_rename_key(ex, old_key, new_key) for ex in train]
    dev   = [_rename_key(ex, old_key, new_key) for ex in dev]
    test  = [_rename_key(ex, old_key, new_key) for ex in test]
    return train, dev, test

def _rename_key(ex:dict, old_key:str='content', new_key:str='text'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex

def _remove_html_tags(train:list, dev:list, test:list):
    train = [_remove_html_tags_ex(ex) for ex in train]
    dev   = [_remove_html_tags_ex(ex) for ex in dev]
    test  = [_remove_html_tags_ex(ex) for ex in test]
    return train, dev, test

def _remove_html_tags_ex(ex:dict):
    CLEANR = re.compile('<.*?>') 
    ex['text'] = re.sub(CLEANR, '', ex['text'])
    return ex

#== Functions to synthetically add shortcuts =============================================================#
def create_len_bias(data_name):
    data_name, _, bias_ratio = data_name.split('_')
    print(bias_ratio)
    train, dev, test = load_single_texts(data_name)
    
    bias_ratio = float(bias_ratio)
    train, dev = [_class_divide(split, bias_ratio) for split in [train, dev]]
    return train, dev, test

def _class_divide(data_split:List[dict], bias_ratio:float=1.0):
    labels = set([x['label'] for x in data_split])
    
    # Split data set by classes, and order by len
    classes_split = {}
    for label in labels:
        class_split = [x for x in data_split if x['label'] == label]
        classes_split[label] = sorted(class_split, key=lambda x: len(x['text']))
    
    # introduce natural length shortcuts in data
    output, unused = [], []
    for k, class_examples in classes_split.items():
        ex_range = len(class_examples)/len(labels)
        r1, r2 = int(k*ex_range), int((k+1)*ex_range)
        selected_ex = class_examples[r1:r2]
        output += selected_ex
        unused += class_examples[:r1]
        unused += class_examples[r2:]
    
    # add other points if
    random.shuffle(unused)
    N = int((1-bias_ratio)*len(unused))
    print(bias_ratio, N)
    print('raw', len(output), len(unused))

    output += unused[:N]
    
    print(len(output))
    return output
    