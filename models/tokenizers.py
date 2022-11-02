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
    elif system == 't5-xl': 
        return T5TokenizerFast.from_pretrained("t5-3b", model_max_length=512)
    elif system == 't5-v1.1-small': 
        return T5TokenizerFast.from_pretrained("google/t5-v1_1-small", model_max_length=512)
    elif system == 't5-v1.1-base': 
        return T5TokenizerFast.from_pretrained("google/t5-v1_1-base", model_max_length=512)
    elif system == 't5-v1.1-large': 
        return T5TokenizerFast.from_pretrained("google/t5-v1_1-large", model_max_length=512)
    elif system == 't5-v1.1-xxl': 
        return T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", model_max_length=512)
                         
    raise ValueError("{} is invalid".format(system))
