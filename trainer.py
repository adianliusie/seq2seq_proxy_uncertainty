import wandb
import torch
import logging
import os

from collections import namedtuple
from types import SimpleNamespace

from .data_utils.data_handler import DataHandler
from .batchers.batcher import Batcher
from .utils.dir_helper import DirHelper
from .models.seq2seq import load_seq2seq_transformer 
from .utils.general import save_json, load_json, no_grad
from .loss import load_model_loss

class Trainer():
    """"base class for finetuning transformer to datasets"""
    def __init__(self, exp_path:str, m_args:namedtuple):
        self.set_up_exp(exp_path, m_args)
        self.set_up_helpers(m_args)

    #== MAIN TRAIN LOOP ===========================================================================#
    def set_up_helpers(self, m_args:namedtuple):
        self.model_args = m_args
        self.data_handler = DataHandler(trans_name=m_args.transformer)
        self.batcher = Batcher(max_len=m_args.max_len)
        self.model = load_seq2seq_transformer(system=m_args.transformer)

    def train(self, args:namedtuple):
        self.save_args('train_args.json', args)
        if args.wandb: self.set_up_wandb(args)
 
        train, dev, test = self.data_handler.prep_data(args.data_set, args.lim)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            eps=args.epsilon
        )
        #scheduler = ...

        self.model_loss = load_model_loss(loss_fn=args.loss_fn, 
                                          model=self.model,
                                          tokenizer=self.data_handler.tokenizer)

        best_metric = {}
        self.device = args.device
        self.to(self.device)
        
        for epoch in range(args.epochs):
            #==  TRAINING =========================================================================#
            self.model.train()
            self.model_loss.reset_metrics()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                loss = self.model_loss(batch)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print train performance every log_every samples
                if k%(args.log_every//args.bsz) == 0:
                    metrics = self.log_metrics(mode='train', step=k, lr=args.lr)
                    if args.wandb: self.log_wandb(metrics, mode='train')

                #==  DEV ==========================================================================#
                if k%(args.val_every//args.bsz) == 0:   
                    self.model_loss.reset_metrics()     
                    dev_b = self.batcher(data=dev, bsz=args.bsz, shuffle=False)
                    for batch in dev_b:
                        self.model_loss.eval_forward(batch)
                    
                    metrics = self.log_metrics(mode='dev')
                    if args.wandb: self.log_wandb(metrics, mode='dev')

                    # save performance if best dev performance 
                    if metrics['loss'] < best_metric.get('loss', 10000):
                        best_metric = metrics.copy()
                        self.save_model()
                    
                    self.log_metrics(mode='best_dev', metrics=best_metric)

    #== LOGGING UTILS =============================================================================#
    def log_metrics(self, mode:str='train', step:int=None, lr:float=None, metrics=None):
        if metrics is None:
            metrics = {key: value.avg for key, value in self.model_loss.metrics.items()}

        if   mode == 'train'   : msg = f'iteration:{step:<6}  lr: {lr:.5f}  ' 
        elif mode == 'dev'     : msg = 'dev       |||  '
        elif mode == 'best_dev': msg = 'best dev  |||  '

        for key, value in metrics.items():
            msg += f'{key}: {value:.3f} '
        self.logger.info(msg)

        self.model_loss.reset_metrics()   
        return metrics

    def log_wandb(self, metrics, mode):
        if mode == 'dev': metrics = {f'dev_{key}':value for key, value in metrics.items()}
        wandb.log(metrics)

    #== FILE UTILS ================================================================================#
    def set_up_exp(self, exp_path:str, m_args:namedtuple):
        self.exp_path = exp_path
        self.logger = self.set_logger()

        self.logger.info("creating experiment folder")
        os.makedirs(self.exp_path)
        os.mkdir(f'{self.exp_path}/models')

        self.save_args('model_args.json', m_args)

    def save_args(self, name:str, data:namedtuple):
        """saves arguments into json format"""
        save_path = f'{self.exp_path}/{name}'
        save_json(data.__dict__, save_path)

    def load_args(self, name:str)->SimpleNamespace:
        args = load_json(f'{self.exp_path}/{name}')
        return SimpleNamespace(**args)
    
    def set_logger(self):
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger

    #== MODEL UTILS  ==============================================================================#
    def save_model(self, name:str='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(
            self.model.state_dict(), 
            f'{self.exp_path}/models/{name}.pt'
        )
        self.model.to(device)

    def load_model(self, name:str='base'):
        self.model.load_state_dict(
            torch.load(f'{self.exp_path}/models/{name}.pt'))

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    #==  WANDB UTILS  =============================================================================#
    def set_up_wandb(self, args:namedtuple):
        wandb.init(project=args.wandb, 
                   entity='mg-speech-group',
                   name=self.dir.exp_name, 
                   group=self.dir.base_name, 
                   dir=self.dir.abs_path, 
                   reinit=True)

        # save experiment config details
        cfg = {}
        cfg['epochs']      = args.epochs
        cfg['bsz']         = args.bsz
        cfg['lr']          = args.lr
        cfg['transformer'] = self.model_args.transformer       
        cfg['data_set']    = args.data_set

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
