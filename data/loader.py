from typing import List, Dict
from tqdm import tqdm

import os

from datasets import load_dataset


__all__ = [
    "load_nmt_data",
]


def load_nmt_data(dname: str = 'wmt16-en-de'):
    if dname == 'wmt16-en-de': 
        return load_wmt16(lang = 'de')
    if dname == 'newscommentary-en-de':
        return load_newscommentary(lang = 'de')
    if dname == 'newstest13':
        return load_newstest13(lang = 'de')
    if dname == 'newstest14':
        return load_newstest14(lang = 'de')
    if dname == 'newstest15':
        return load_newstest15(lang = 'de')
    if dname == 'newstest16':
        return load_newstest16(lang = 'de')
    if dname == 'newstest17':
        return load_newstest17(lang = 'de')
    if dname == 'newstest18':
        return load_newstest18(lang = 'de')
    if dname == 'newstest19':
        return load_newstest19(lang = 'de')
    if dname == 'newstest20':
        return load_newstest20(lang = 'de')
    
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


def format_newstest(src, ref):
    output = []
    for (src_sent, ref_sent) in zip(src, ref):
        output.append({
            'input_text': src_sent.strip("\n"), 
            'label_text': ref_sent.strip("\n"),
        })
    return output


def load_newstest13(lang: str = 'de'):
    """
    Loads newstest-13 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2013-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2013-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest14(lang: str = 'de'):
    """
    Loads newstest-14 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2014-deen-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2014-deen-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest15(lang: str = 'de'):
    """
    Loads newstest-15 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2015-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2015-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest16(lang: str = 'de'):
    """
    Loads newstest-16 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2016-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2016-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest17(lang: str = 'de'):
    """
    Loads newstest-17 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2017-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2017-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest18(lang: str = 'de'):
    """
    Loads newstest-18 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2018-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2018-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest19(lang: str = 'de'):
    """
    Loads newstest-19 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2019-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2019-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest20(lang: str = 'de'):
    """
    Loads newstest-20 en-de data from huggingface
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, "newstest2020-ende-src.en.sgm.text"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "newstest2020-ende-ref.de.sgm.text"), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output