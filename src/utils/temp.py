# Adian Liusie, 2022-SEP-02

import torch
from typing import Callable
from transformers import ElectraTokenizerFast, ElectraModel 
from transformers import BertTokenizerFast, BertModel, BertConfig
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import AutoTokenizer, AutoModel

#== General Util Methods ===================================================================================
def no_grad(func:Callable)->Callable:
    """ decorator which detaches gradients """
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner

#== Methods for loading tokenizer ==========================================================================
def load_tokenizer(system:str)->'Tokenizer':
    """ downloads and returns the relevant pretrained tokenizer from huggingface """
    if system   == 'bert'          : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_rand'     : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_large'    : tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    elif system == 'bert_tiny'     : tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta'       : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'electra'       : tokenizer = ElectraTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'electra_large' : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return tokenizer


#== Methods for loading the base transformers ==============================================================
def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert'       : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert_rand'  : trans_model = BertModel(BertConfig())
    elif system == 'bert_large' : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'bert_tiny'  : trans_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta'    : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'electra'    : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra_large':
        trans_model= ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model
