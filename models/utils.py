def set_dropout(model, dropout_rate):
    for layer in model.encoder.layer[::-1][:num_rand_layers]:
        
        layer.apply(trans_model._init_weights)