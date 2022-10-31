from typing import List, Dict
from tqdm import tqdm

from datasets import load_dataset


__all__ = [
    "load_nmt_data",
]


def load_nmt_data(dname: str = 'wmt16-en-de'):
    if dname == 'wmt16-en-de': 
        return load_wmt16(lang = 'de')
    if dname == 'newscommentary-en-de':
        return load_newscommentary(lang = 'de')
    
    raise ValueError("Invalid dataset name: {}".format(dname))
                       

def format_wmt16(data: List[Dict], lang: str) -> List[Dict]:
    """
    Converts data split on huggingface into format for the framework
    """
    output = []
    for ex in tqdm(data):
        ex = ex['translation']
        output.append({
            'input_text': ex['en'], 
            'label_text': ex[lang],
        })
    return output


def load_wmt16(lang: str = 'de'):
    """
    Loads wmt16-de-en data from huggingface
    """
    data = load_dataset("wmt16", "{}-en".format(lang))
    train, dev, test = data['train'], data['validation'], data['test']
    train, dev, test = [format_wmt16(split, lang) for split in [train, dev, test]]
    return train, dev, test


def load_newscommentary(lang: str = 'de'):
    """
    Loads news_commentary-de-en data from huggingface
    """
    data = load_dataset(f"news_commentary", "{}-en".format(lang))
    adata = load_dataset(f"wmt16", "{}-en".format(lang))
    train, dev, test = data['train'], adata['validation'], adata['test']
    train, dev, test = [format_wmt16(split, lang) for split in [train, dev, test]]
    return train, dev, test