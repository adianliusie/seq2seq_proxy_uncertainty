import os
import logging
from collections import namedtuple
from types import SimpleNamespace

import wandb
import torch

from data.handler import DataHandler
from batcher import Batcher
from models.transformer import load_seq2seq_transformer 
from utils.general import save_json, load_json
from loss import load_loss


class Trainer(object):
    """
    Base class for finetuning transformer to datasets
    """
    def __init__(self, path: str, args: namedtuple):
        self.setup_exp(path, args)
        self.setup_helpers(args)

    def setup_helpers(self, args: namedtuple):
        self.model_args = args
        self.data_handler = DataHandler(name=args.transformer)
        self.batcher = Batcher(maxlen=args.maxlen)
        self.model = load_seq2seq_transformer(system=args.transformer)

    def train(self, args:namedtuple):
        self.save_args('train_args.json', args)
        if args.wandb: 
            self.setup_wandb(args)
 
        train, dev, test = self.data_handler.prep_data(args.data_set, args.lim)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            eps=args.epsilon
        )
        #scheduler = ...

        self.model_loss = load_loss(
            loss=args.loss,
            args=args,
            model=self.model,
        )

        best_metric = {}
        self.device = args.device
        self.to(self.device)
        
        for epoch in range(args.epochs):

            self.model.train()
            self.model_loss.reset_metrics()
            trainloader = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for step, batch in enumerate(trainloader, start=1):
                loss = self.model_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # Print train performance every log_every samples
                if step % (args.log_every // args.bsz) == 0:
                    metrics = self.log_metrics(mode='train', step=step, lr=args.lr)
                    if args.wandb: 
                        self.log_wandb(metrics, mode='train')

                # Validation performance
                if step % (args.val_every//args.bsz) == 0:   
                    self.model_loss.reset_metrics()     
                    devloader = self.batcher(data=dev, bsz=args.bsz, shuffle=False)
                    for batch in devloader:
                        self.model_loss.eval_forward(batch)
                    
                    metrics = self.log_metrics(mode='dev')
                    if args.wandb: 
                        self.log_wandb(metrics, mode='dev')

                    # Save performance if best dev performance 
                    if metrics['loss'] < best_metric.get('loss', float('inf')):
                        best_metric = metrics.copy()
                        self.save_model()
                    
                    self.log_metrics(mode='best_dev', metrics=best_metric)

    def log_metrics(self, mode: str = 'train', step: int = None, lr: float = None, metrics = None):
        if metrics is None:
            metrics = {key: value.avg for key, value in self.model_loss.metrics.items()}

        if mode == 'train': 
            msg = 'iteration:{:<6}  lr: {:.5f}  '.format(step, lr)
        elif mode == 'dev': 
            msg = 'dev       |||  '
        elif mode == 'best_dev': 
            msg = 'best dev  |||  '

        for key, value in metrics.items():
            msg += '{}: {:.3f} '.format(key, value)
        self.logger.info(msg)

        self.model_loss.reset_metrics()   
        return metrics

    def log_wandb(self, metrics, mode):
        if mode == 'dev': 
            metrics = {'dev_{}'.format(key): value for key, value in metrics.items()}
        wandb.log(metrics)

    def setup_exp(self, exp_path: str, args: namedtuple):
        self.exp_path = exp_path
        self.logger = self.set_logger()

        self.logger.info("Creating experiment folder")
        os.makedirs(self.exp_path)
        os.mkdir(os.path.join(self.exp_path, 'models'))

        self.save_args('model_args.json', args)

    def save_args(self, name: str, data: namedtuple):
        """
        Saves arguments into json format
        """
        path = os.path.join(self.exp_path, name)
        save_json(data.__dict__, path)

    def load_args(self, name : str) -> SimpleNamespace:
        path = os.path.join(self.exp_path, name)
        args = load_json(path)
        return SimpleNamespace(**args)
    
    def set_logger(self):
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger

    def save_model(self, name : str = 'base'):
        # Get current model device
        device = next(self.model.parameters()).device
        
        # Save in cpu mode
        self.model.to("cpu")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.exp_path, 'models', '{}.pt'.format(name))
        )

        # Return to original device
        self.model.to(device)

    def load_model(self, name: str = 'base'):
        self.model.load_state_dict(
            os.path.join(self.exp_path, 'models', '{}.pt'.format(name))
        )

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    def setup_wandb(self, args: namedtuple):
        wandb.init(
            project=args.wandb, 
            entity='mg-speech-group',
            name=self.dir.exp_name, 
            group=self.dir.base_name, 
            dir=self.dir.abs_path, 
            reinit=True
        )

        # save experiment config details
        cfg = {
            'epochs': args.epochs,
            'bsz': args.bsz,
            'lr': args.lr,
            'transformer': self.model_args.transformer,
            'dataset': args.data_set,
        }

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
