from transformers import PreTrainedTokenizer
from transformers import T5TokenizerFast
from transformers import BertTokenizerFast, RobertaTokenizerFast, ElectraTokenizerFast

SEQ2SEQ_TOKENIZERS = ['t5_small', 't5_base', 't5_large']
def load_seq2seq_tokenizer(system:str)->PreTrainedTokenizer:
    """ downloads and returns the relevant pretrained seq2seq tokenizer from huggingface """
    if   system == 't5_small' : tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    elif system == 't5_base'  : tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    elif system == 't5_large' : tokenizer = T5TokenizerFast.from_pretrained("t5-large")
    else: raise ValueError(f"{system} is invalid: must be one of {SEQ2SEQ_TOKENIZERS}")
    return tokenizer

ENCODER_TOKENIZERS = ['bert', 'bert_large', 'roberta' , 'roberta_large', 'electra_base', 'electra_large']
def load_encoder_tokenizer(system:str)->PreTrainedTokenizer:
    if system   == 'bert'          : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_large'    : tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    elif system == 'roberta'       : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'roberta_large' : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    elif system == 'electra_base'  : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
    elif system == 'electra_large' : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    else: raise ValueError(f"{system} is invalid: must be one of {ENCODER_TOKENIZERS}")
    return tokenizer

def load_tokenizer(system:str)->PreTrainedTokenizer:
    if   system in SEQ2SEQ_TOKENIZERS:  tokenizer = load_seq2seq_tokenizer(system)
    elif system in ENCODER_TOKENIZERS:  tokenizer = load_encoder_tokenizer(system)
    return tokenizer