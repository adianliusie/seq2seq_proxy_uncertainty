from .cross_entropy_loss import CrossLoss

def load_model_loss(loss_fn, model, tokenizer=None, args=None):
    if loss_fn == 'cross_entropy':
        return CrossLoss(model, tokenizer)