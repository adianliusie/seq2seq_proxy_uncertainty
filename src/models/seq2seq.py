from transformers import T5ForConditionalGeneration

def load_seq2seq_transformer(system:str):
    """ downloads and returns the relevant pretrained seq2seq transformer from huggingface """
    if   system == 't5_small' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    elif system == 't5_base'  : trans_model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    elif system == 't5_large' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-large", return_dict=True)
    else: raise ValueError(f"{system} is an invalid system: no seq2seq model found")
    return trans_model

