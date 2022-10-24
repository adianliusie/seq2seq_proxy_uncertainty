import random

from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset
from tqdm import tqdm

#== Main loading function ==============================================================================# 
class TextPair(TypedDict):
    """Output example formatting (only here for documentation)"""
    input_text : str
    label_text : str

def load_nmt_data(data_name):
    if data_name   == 'wmt16-de': train, dev, test = load_wmt16(lang='de')
    else: raise ValueError(f"invalid dataset name: {data_name}")
    return train, dev, test
                       
#== NMT data set loader ===================================================================================#
def load_wmt16(lang:str='de'):
    """ loads wmt16 data from huggingface """
    data = load_dataset("wmt16", f"{lang}-en")
    train, dev, test = data['train'], data['validation'], data['test']
    train, dev, test = [format_wmt16(split, lang) for split in [train, dev, test]]
    return train, dev, test
    
def format_wmt16(data:List[dict], lang:str):
    """ converts data split on huggingface into format for the framework """
    output = []
    for ex in tqdm(data):
        ex_data = ex['translation']
        output.append({'input_text':ex_data['en'], 'label_text':ex_data[lang]})
        if len(output) > 10_000: break # temp line to make dataset smaller
    return output
 