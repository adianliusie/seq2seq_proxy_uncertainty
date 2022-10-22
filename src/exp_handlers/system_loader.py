import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from typing import List
from functools import lru_cache

from .trainer import Trainer
from ..batchers.batcher import Batcher
from ..utils.torch_utils import no_grad
from ..utils.dir_helper import DirHelper
from ..data_utils.data_loader import DataLoader
from ..utils.evaluation import calc_acc, anneal_probs

#== Class to load trained system and generate predictions ===================================#
class SystemLoader(Trainer):
    """Base loader class- the inherited class inherits
       the Trainer so has all experiment methods"""

    #== 'Set up Methods' to load the given path =============================================#
    def __init__(self, exp_path:str, device=None):
        self.exp_path = exp_path
        self.dir = DirHelper.load_dir(exp_path)
        self.device = device
        
    def set_up_helpers(self):
        #load training arguments and set up helpers
        args = self.dir.load_args('model_args.json')
        super().set_up_helpers(args)

        #load final model
        self.load_model()
        self.model.eval()
        
        #set device
        if not getattr(self, 'device', None): self.device = 'cuda:0'
        self.to(self.device)

    #== Methods to load predictions/probabilities for given dataset =========================#
    def load_preds(self, data_name:str, mode:str)->dict:
        probs = self.load_probs(data_name, mode)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
    def load_probs(self, data_name:str, mode:str, calibrate=False)->dict:
        """loads predictions if saved, else generates"""
        if not self.dir.probs_exists(data_name, mode):
            self.set_up_helpers()
            self.generate_probs(data_name, mode)
        probs = self.dir.load_probs(data_name, mode)
        
        if calibrate:
            labels = self.load_labels(data_name, mode)
            probs  = anneal_probs(probs, labels, silent=False)
            
        return probs

    def generate_probs(self, data_name:str, mode:str):
        probabilties = self._probs(data_name, mode)
        self.dir.save_probs(probabilties, data_name, mode)

    @no_grad
    def _probs(self, data_name:str, mode:str='test'):
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_loader.prep_split(data_name, mode)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        
        probabilties = {}
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model_output(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probabilties[ex_id] = prob.cpu().numpy()
        return probabilties
  
    #== Methods to get hidden layer outputs =================================================#
    @lru_cache(maxsize=5)
    @no_grad
    def get_last_layer_repr(self, data_name:str=None, mode:str='test', lim:int=None):
        #load model if not currently there:
        if not hasattr(self, 'model'):
            self.set_up_helpers()
            self.device = 'cuda'
            
        #prepare data for cases where data_name is given
        data = self.data_loader.prep_split(data_name, mode=mode, lim=lim)

        #create batches
        eval_batches = self.batcher(data=data, bsz=1, shuffle=False)
        
        #get output vectors
        self.to(self.device)
        self.model.eval()
        output_dict = {}
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            model_output = self.model_output(batch)
            output_dict[ex_id] = model_output.h.squeeze(0).cpu().numpy()
            
        return output_dict

    def get_cached_last_layer(self, data_name:str=None, mode:str='test', lim:int=None):
        """temp function that caches hidden repr calculations"""
        from ..utils.general import save_pickle, load_pickle
        CACHE_DIR  = '/home/al826/rds/hpc-work/2022/shortcuts/layer_training/investigations/shortcut_probe/cached_h'
        file_name  = f"{self.exp_path.replace('/', '-')}_{data_name}_{mode}.json"
        cache_path = f"{CACHE_DIR}/{file_name}"
       
        if os.path.isfile(cache_path):
            output = load_pickle(cache_path)
        else:
            output = self.get_last_layer_repr(data_name, mode)
            save_pickle(output, cache_path)
        
        return output
        
    #== Util Methods to enable easy access to data utils ====================================#
    @staticmethod
    def load_labels(data_name:str, mode:str='test', lim=None)->dict:
        eval_data = DataLoader.load_split(data_name, mode)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[ex.ex_id] = ex.label
        return labels_dict

    @staticmethod
    def load_inputs(data_name:str, mode:str='test')->dict:
        eval_data = DataLoader.load_split(data_name, mode)
        inputs_dict = {}
        for ex in eval_data:
            inputs_dict[ex.ex_id] = ex
        return inputs_dict
    
    @staticmethod
    def get_eval_data(data_name:str, mode:str='test'):
        return self.data_loader.prep_MCRC_split(data_name, mode)
    
    #== Util Methods to enable easy access to evaluation utils ==============================#
    def calc_acc(self, data_name, mode):
        preds  = self.load_preds(data_name, mode)
        labels = self.load_labels(data_name, mode)
        return calc_acc(preds=preds, labels=labels)
    
#== Wrapper for convenient Ensemple predictions ============================================#
class EnsembleLoader(SystemLoader):
    def __init__(self, exp_path:str, device=None):
        self.exp_path = exp_path
        self.paths  = [f'{exp_path}/{seed}' for seed in os.listdir(exp_path) if os.path.isdir(f'{exp_path}/{seed}')]
        self.seeds  = [SystemLoader(seed_path, device) for seed_path in sorted(self.paths)]
    
    def load_probs(self, data_name:str, mode)->dict:
        seed_probs = [seed.load_probs(data_name, mode) for seed in self.seeds]

        ex_ids = seed_probs[0].keys()
        assert all([i.keys() == ex_ids for i in seed_probs])

        ensemble = {}
        for ex_id in ex_ids:
            probs = [seed[ex_id] for seed in seed_probs]
            probs = np.mean(probs, axis=0)
            ensemble[ex_id] = probs
        return ensemble    
    
    def load_preds(self, data_name:str, mode)->dict:
        probs = self.load_probs(data_name, mode)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
    #== Util Methods to enable easy access to evaluation utils ==============================#
    def calc_ens_acc(self, data_name, mode):
        accs = [seed.calc_acc(data_name, mode) for seed in self.seeds]
        mean, std = np.mean(accs), np.std(accs)
        return round(mean, 2), round(std, 2)

    def get_cached_last_layer(self, data_name:str=None, mode:str='test', lim:int=None, data:List=None):
        seed_1 = self.seeds[0]
        output = seed_1.get_cached_last_layer(data_name, mode)
        return output
    
        