import os
import logging

from itertools import cycle
from collections import namedtuple
from types import SimpleNamespace

import wandb
import torch

from data.handler import DataHandler
from batcher import Batcher
from models.models import load_model 
from utils.general import save_json, load_json
from loss import load_loss


# Creat Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.model = load_model(system=args.transformer)

    def train(self, args: namedtuple):

        # Save arguments for future reference and quick loading
        self.save_args('train-args.json', args)

        # Setup wandb for online tracking of experiments
        if args.wandb: 
            self.setup_wandb(args)
 
        # Get train, val, test split of data
        train, dev, test = self.data_handler.prep_data(args.dataset, args.datasubset)

        # All experiments will use the adam-w optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr,
        )

        # We use a triangular finetuning schedule
        def lr_lambda(step):
            if step < args.num_warmup_steps:
                return step / args.num_warmup_steps
            return (args.num_steps - step) / (args.num_steps - args.num_warmup_steps)
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )

        # The the loss function which takes in the model
        self.model_loss = load_loss(
            loss=args.loss,
            args=args,
            model=self.model,
        )

        # Store the best metrics shere
        self.best_metric = {'dev': {}, 'test': {}}

        # Device management
        self.to(args.device)

        # Setup model for translation
        self.model.train()

        # Reset loss metrics
        self.model_loss.reset_metrics()

        # Create batched dataset
        trainloader = self.batcher(data = train, numtokens = args.num_tokens, shuffle = True)

        for step, batch in enumerate(cycle(trainloader), start = 0):

            # Reset optimizer every n steps
            if step % args.num_gradient_accum == 0:
                optimizer.zero_grad()
            
            # Perform forward pass
            loss = self.model_loss(batch) / args.num_gradient_accum

            # Perform backward pass
            loss.backward()

            # Update optimizer and scheduler learning rates
            if (step + 1) % args.num_gradient_accum == 0:
                optimizer.step()
                scheduler.step()


            # Print train performance every log_every samples
            if step % (args.log_every * args.num_gradient_accum) == 0:
                metrics = self.log_metrics(
                    mode = 'train', 
                    step = (step + 1) // args.num_gradient_accum, 
                    lr = scheduler.get_last_lr()[0],
                )
                self.log_wandb(args, metrics, mode='train')


            # Validation performance
            if step % (args.val_every * args.num_gradient_accum) == 0:  
                self.validate(args, dev, mode = 'dev')
                self.validate(args, test, mode = 'test')

            # Stop training at breaking point
            if step >= args.num_steps * args.num_gradient_accum:
                break

    def validate(self, args, data, mode):

        # Ensure correct mode
        assert mode in ['dev', 'test']

        # Reset metrics for dev performance
        self.model_loss.reset_metrics()

        # Create new dev dataset  
        loader = self.batcher(
            data = data, 
            numtokens = args.num_tokens, 
            shuffle = False
        )

        # Save metrics for all the dataset
        for batch in loader:
            self.model_loss.eval_forward(batch)
        
        # Record metrics
        metrics = self.log_metrics(mode = mode)
        self.log_wandb(args, metrics, mode = mode)

        # Save performance if best dev performance 
        if metrics['loss'] < self.best_metric[mode].get('loss', float('inf')):
            self.best_metric[mode] = metrics.copy()
            self.save_model()
        
        # Best dev performance so far
        self.log_metrics(mode=f'best-{mode}', metrics=self.best_metric[mode])


    def log_metrics(self, mode: str = 'train', step: int = None, lr: float = None, metrics = None):
        if metrics is None:
            metrics = {key: value.avg for key, value in self.model_loss.metrics.items()}

        if mode == 'train': 
            msg = 'iteration:{:<6}  lr: {:.5f}  '.format(step, lr)
        elif mode == 'dev': 
            msg = 'dev       |||  '
        elif mode == 'best-dev': 
            msg = 'best dev  |||  '
        elif mode == 'test': 
            msg = 'test      |||  '
        elif mode == 'best-test': 
            msg = 'best test |||  '

        for key, value in metrics.items():
            msg += '{}: {:.3f} '.format(key, value)
        logger.info(msg)

        self.model_loss.reset_metrics()   
        return metrics

    def log_wandb(self, args, metrics, mode):
        if not args.wandb:
            return 
        if mode == 'dev': 
            metrics = {'dev_{}'.format(key): value for key, value in metrics.items()}
        wandb.log(metrics)

    def setup_exp(self, exp_path: str, args: namedtuple):
        self.exp_path = exp_path

        if not os.path.isdir(self.exp_path):
            logger.info("Creating experiment folder")
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
        group_name = self.exp_path.split('-v')[0]
        wandb.init(
            project='proxy-uncertainty-{}'.format(args.dataset),
            entity='mg-speech-group',
            name=self.exp_path, 
            group=group_name 
        )

        # save experiment config details
        cfg = {
            'dataset': args.dataset,
            'num_tokens': args.num_tokens,
            'lr': args.lr,
            'transformer': self.model_args.transformer,
            'dataset': args.dataset,
        }

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
