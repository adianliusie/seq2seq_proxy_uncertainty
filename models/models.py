from transformers import T5ForConditionalGeneration


__all__ = [
    "load_model"
]


def load_model(system: str) -> T5ForConditionalGeneration:
    """
    Downloads and returns the relevant pretrained seq2seq transformer from huggingface
    """
    if system == 't5-small':
        return T5ForConditionalGeneration.from_pretrained("t5-small", return_dict = True)
    elif system == 't5-base': 
        return T5ForConditionalGeneration.from_pretrained("t5-base", return_dict = True)
    elif system == 't5-large': 
        return T5ForConditionalGeneration.from_pretrained("t5-large", return_dict = True)
    elif system == 't5-xl': 
        return T5ForConditionalGeneration.from_pretrained("t5-3b", return_dict = True)
    if system == 't5-v1.1-small':
        return T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small", return_dict = True)
    if system == 't5-v1.1-base':
        return T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base", return_dict = True)
    if system == 't5-v1.1-large':
        return T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large", return_dict = True)
    if system == 't5-v1.1-xl':
        return T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", return_dict = True)

    raise ValueError("{} is an invalid system: no seq2seq model found".format(system))
