import wandb
import torch

from collections import namedtuple

from ..data_utils.data_handler import DataHandler
from ..batchers.batcher import Batcher
from ..utils.dir_helper import DirHelper
from ..models.seq2seq import load_seq2seq_transformer 
from ..utils.general import no_grad
from ..losses import load_model_loss

class Trainer():
    """"base class for finetuning transformer to datasets"""
    def __init__(self, exp_name:str, m_args:namedtuple):
        self.dir = DirHelper(exp_name)
        self.dir.save_args('model_args.json', m_args)
        self.set_up_helpers(m_args)
        
    #== MAIN TRAIN LOOP ==============================================================#
    def set_up_helpers(self, m_args:namedtuple):
        self.model_args = m_args
        self.data_loader = DataHandler(trans_name=m_args.transformer)
        self.batcher = Batcher(max_len=m_args.max_len)
        self.model = load_seq2seq_transformer(system=m_args.transformer)

    def train(self, args:namedtuple):
        self.dir.save_args('train_args.json', args)
        if args.wandb: self.set_up_wandb(args)
 
        train, dev, test = self.data_loader.prep_data(args.data_set, args.lim)

        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                      lr=args.lr, eps=args.epsilon)
        self.model_loss = load_model_loss(args.loss_fn, self.model)

        best_epoch = (-1, 10000)
        self.device = args.device
        self.to(self.device)
        
        for epoch in range(args.epochs):
            #==  TRAINING ==============================================#
            self.model.train()
            self.dir.reset_metrics()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                output = self.model_loss(batch)

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                # accuracy logging
                self.dir.update_avg_metrics(loss=output.loss)

                # print train performance every now and then
                if k%(args.print_len//args.bsz) == 0:
                    perf = self.dir.print_perf('train', epoch, k)
                    if args.wandb:
                         wandb.log({'epoch':epoch, 'loss':perf.loss})
            
            # save performance if best dev performance 
            if perf.loss < best_epoch[1]:
                best_epoch = (epoch, perf.loss)
                self.save_model()
            
            self.dir.log(f'best dev epoch: {best_epoch}')
            
            if epoch - best_epoch[0] >= 3:
                break
        
        #== Final TEST performance ==================================#
        self.load_model()
        test_perf = self.system_eval(test, epoch, mode='test')
    
    #== EVAL METHODS ================================================================#
    @no_grad
    def system_eval(self, data, epoch:int, mode='dev'):
        self.dir.reset_metrics()         
        batches = self.batcher(data=data, bsz=1, shuffle=False)
        for k, batch in enumerate(batches, start=1):
            output = self.model_loss(batch)
            self.dir.update_avg_metrics(loss=output.loss)
        perf = self.dir.print_perf(mode, epoch, 0)
        return perf

    #== MODEL UTILS  ================================================================#
    def save_model(self, name:str='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.abs_path}/models/{name}.pt')
        self.model.to(device)

    def load_model(self, name:str='base'):
        self.model.load_state_dict(
            torch.load(self.dir.abs_path + f'/models/{name}.pt'))

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    #==  WANDB UTILS  ===============================================================#
    def set_up_wandb(self, args:namedtuple):
        wandb.init(project=args.wandb, entity="adian",
                   name=self.dir.exp_name, group=self.dir.base_name, 
                   dir=self.dir.abs_path, reinit=True)

        # save experiment config details
        cfg = {}
        cfg['epochs']      = args.epochs
        cfg['bsz']         = args.bsz
        cfg['lr']          = args.lr
        cfg['transformer'] = self.model_args.transformer       
        cfg['data_set']    = args.data_set

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
