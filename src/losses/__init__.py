from .cross_entropy_loss import BaseCrossLoss

def load_model_loss(loss_name, model, args=None):
    if loss_name == 'cross_entropy':
        return BaseCrossLoss(model)