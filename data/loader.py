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
    if dname == 'iwslt2014':
        return load_iwslt201x(lang = 'de', year = '2014')
    if dname == 'iwslt2015':
        return load_iwslt201x(lang = 'de', year = '2015')
    if dname == 'iwslt2016':
        return load_iwslt201x(lang = 'de', year = '2016')
    if dname == 'iwslt2017':
        return load_iwslt2017(lang = 'de')
    if dname == 'khresmoi-dev':
        return load_khresmoi_dev(lang = 'de')
    if dname == 'khresmoi-test':
        return load_khresmoi_test(lang = 'de')
    if dname == 'mtnt2019':
        return load_mtnt2019(lang = 'ja')
    if dname == 'jesc':
        return load_jesc(lang = 'ja')
    if dname == 'kftt':
        return load_kftt(lang = 'ja')
    if dname == 'ted':
        return load_ted(lang = 'ja')

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


def load_newstest(path_src: str, path_ref: str):
    """
    Loads generic newstest en-de data from file locally
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-newstest/dev/text"
    with open(os.path.join(path, path_src), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, path_ref), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_newstest13(lang: str = 'de'):
    """
    Loads newstest-13 en-de data from file locally
    """
    return load_newstest(
        "newstest2013-src.en.sgm.text",
        "newstest2013-ref.de.sgm.text",
    )


def load_newstest14(lang: str = 'de'):
    """
    Loads newstest-14 en-de data from file locally
    """
    return load_newstest(
        "newstest2014-deen-src.en.sgm.text",
        "newstest2014-deen-ref.de.sgm.text",
    )


def load_newstest15(lang: str = 'de'):
    """
    Loads newstest-15 en-de data from file locally
    """
    return load_newstest(
        "newstest2015-ende-src.en.sgm.text",
        "newstest2015-ende-ref.de.sgm.text",
    )


def load_newstest16(lang: str = 'de'):
    """
    Loads newstest-16 en-de data from file locally
    """
    return load_newstest(
        "newstest2016-ende-src.en.sgm.text",
        "newstest2016-ende-ref.de.sgm.text",
    )


def load_newstest17(lang: str = 'de'):
    """
    Loads newstest-17 en-de data from file locally
    """
    return load_newstest(
        "newstest2017-ende-src.en.sgm.text",
        "newstest2017-ende-ref.de.sgm.text",
    )


def load_newstest18(lang: str = 'de'):
    """
    Loads newstest-18 en-de data from file locally
    """
    return load_newstest(
        "newstest2018-ende-src.en.sgm.text",
        "newstest2018-ende-ref.de.sgm.text",
    )


def load_newstest19(lang: str = 'de'):
    """
    Loads newstest-19 en-de data from file locally
    """
    return load_newstest(
        "newstest2019-ende-src.en.sgm.text",
        "newstest2019-ende-ref.de.sgm.text",
    )


def load_newstest20(lang: str = 'de'):
    """
    Loads newstest-20 en-de data from file locally
    """
    return load_newstest(
        "newstest2020-ende-src.en.sgm.text",
        "newstest2020-ende-ref.de.sgm.text",
    )


def load_iwslt201x(lang: str = 'de', year: str = '2014'):
    """
    Loads iswlt-2017 en-de data from huggingface
    """
    dataset = load_dataset("ted_talks_iwslt", language_pair=("en", lang), year=year)
    dataset = dataset['train']
    return format_wmt16(dataset, lang=lang)


def load_iwslt2017(lang: str = 'de'):
    """
    Loads iswlt-2017 en-de data from huggingface
    """
    dataset = load_dataset("iwslt2017", f"iwslt2017-en-{lang}")
    dataset = dataset['test']
    return format_wmt16(dataset, lang=lang)


def load_khresmoi_dev(lang: str = 'de'):
    """
    Loads the Khresmoi Summary development dataset
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-khresmoi/khresmoi-summary-test-set-2.0"
    with open(os.path.join(path, "khresmoi-summary-dev.en"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "khresmoi-summary-dev.de"), "r") as f:
        ref = f.readlines()
    
    output = format_newstest(src, ref)
    return output


def load_khresmoi_test(lang: str = 'de'):
    """
    Loads the Khresmoi Summary development dataset
    """
    path = "/rds/project/rds-8YSp2LXTlkY/data/nmt-khresmoi/khresmoi-summary-test-set-2.0"
    with open(os.path.join(path, "khresmoi-summary-test.en"), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, "khresmoi-summary-test.de"), "r") as f:
        ref = f.readlines()
    
    output = format_newstest(src, ref)
    return output


def load_enja(path_src: str, path_ref: str):
    """
    Loads generic en-ja data from file locally
    """
    path = "/home/yf286/rds/rds-altaslp-8YSp2LXTlkY/data/nmt-enja"
    with open(os.path.join(path, path_src), "r") as f:
        src = f.readlines()
    with open(os.path.join(path, path_ref), "r") as f:
        ref = f.readlines()

    output = format_newstest(src, ref)
    return output


def load_mtnt2019(lang: str = 'ja'):
    """
    Loads mtnt-2019 en-ja data from file locally
    """
    return load_enja(
        "mtnt2019.en-ja.final.en",
        "mtnt2019.en-ja.final.ja",
    )


def load_jesc(lang: str = 'ja'):
    """
    Loads jesc en-ja data from file locally
    """
    return load_enja(
        "test.jesc.en",
        "test.jesc.ja",
    )


def load_kftt(lang: str = 'ja'):
    """
    Loads kftt en-ja data from file locally
    """
    return load_enja(
        "test.kftt.en",
        "test.kftt.ja",
    )


def load_ted(lang: str = 'ja'):
    """
    Loads ted en-ja data from file locally
    """
    return load_enja(
        "test.ted.en",
        "test.ted.ja",
    )
