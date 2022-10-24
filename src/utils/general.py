import os
import json
import pickle
import sys
import torch

from typing import Callable
from pathlib import Path

#== File handling functions ================================================
def save_json(data:dict, path:str):
    with open(path, "x") as outfile:
        json.dump(data, outfile, indent=2)

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def save_pickle(data, path:str):
    with open(path, 'xb') as output:
        pickle.dump(data, output)

def load_pickle(path:str):
    with open(path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

#== Location utils ==========================================================
def join_paths(base_path:str, relative_path:str):
    path = os.path.join(base_path, relative_path)
    path = str(Path(path).resolve()) #convert base/x/x/../../src to base/src
    return path 

def get_base_dir():
    """automatically gets root dir of framework"""
    #gets path of the src folder 
    cur_path = os.path.abspath(__file__)
    src_path = cur_path.split('/src')[0] + '/src'
    base_path = src_path.split('/src')[0]    

    #can be called through a symbolic link, if so go out one more dir.
    if os.path.islink(src_path):
        symb_link = os.readlink(src_path)
        src_path = join_paths(base_path, symb_link)
        base_path = src_path.split('/src')[0]    
        
    return base_path

#== torch utils =============================================================
def no_grad(func:Callable)->Callable:
    """ decorator which detaches gradients """
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner

def get_device(gpu = True):
    """check if device is wanted and available"""
    available = gpu and torch.cuda.is_available()
    return torch.device('cuda' if available else 'cpu')
    
#== Logging utils ===========================================================
def save_script_args():
    CMD = f"python {' '.join(sys.argv)}\n"
    with open('CMDs', 'a+') as f:
        f.write(CMD)
       
