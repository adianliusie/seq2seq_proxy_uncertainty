from transformers import PreTrainedTokenizer
from transformers import T5TokenizerFast

__all___ = [
    "load_tokenizer"
]


def load_tokenizer(system: str) -> PreTrainedTokenizer:
    """
    Downloads and returns the relevant pretrained seq2seq tokenizer from huggingface
    """
    if system == 't5-small': 
        return T5TokenizerFast.from_pretrained("t5-small", model_max_length=512)
    elif system == 't5-base': 
        return T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)
    elif system == 't5-large': 
        return T5TokenizerFast.from_pretrained("t5-large", model_max_length=512)
    raise ValueError("{} is invalid".format(system))
