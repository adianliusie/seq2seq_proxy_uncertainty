from transformers import BertModel, RobertaModel, ElectraModel 

def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert'         : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert_large'   : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'roberta'      : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'electra'      : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra_large': trans_model = ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model
