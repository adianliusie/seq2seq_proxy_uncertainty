import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from scipy import special, stats

#== Basic util functions =========================================================================================#

def probs_to_pred(probs:np.ndarray)->np.ndarray:
    return np.argmax(probs, axis=1)

def get_hits(preds:np.ndarray, labels:np.ndarray):
    return (preds == labels)

def calc_entropy(probs:np.ndarray)->np.ndarray:
    return stats.entropy(probs, base=2, axis=1)

def calc_kl_divergence(probs_1:np.ndarray, probs_2:np.ndarray)->np.ndarray:
    kl = special.rel_entr(probs_1, probs_2)
    kl = kl.sum(axis=1)
    return kl

def calc_shortcut_choice_score(probs:np.ndarray)->np.ndarray:
    return 2**_calc_entropy(probs)

def calc_mutual_information(probs_1:np.ndarray, probs_2:np.ndarray)->np.ndarray:
    entropy_1 = calc_entropy(probs_1)
    entropy_2 = calc_entropy(probs_2)
    mutual_info = entropy_2 - entropy_1
    return mutual_info

def sort_by_entropy(probs:np.ndarray)->List[int]:
    entropies = calc_entropy(probs)
    choices = 2**entropies
    indices = sorted(enumerate(choices), key=lambda x: x[1])
    return indices

#== Conversion util functions ====================================================================================#

def convert_dict_np(input_dict:Dict[str, np.ndarray])->np.ndarray:
    output = []
    for k, p in sorted(input_dict.items()):
        output.append(p)
    output = np.array(output)
    return output
    
def anneal_probs(probs:np.ndarray, labels:np.ndarray):
    #get current model probs and avg max prob
    logits = np.log(probs)
    max_probs = [max(i) for i in probs]
    avg_prob = np.mean(max_probs)
    
    #look at current model accuracy
    preds = probs_to_pred(probs)
    hits = get_hits(preds, labels)
    acc = np.mean(hits)
    
    #do the annealing
    a = 1
    while avg_prob > acc:  
        a += 0.001
        annealed_logits = logits/a
        probs  = special.softmax(annealed_logits, axis=1)
        max_probs = [max(i) for i in probs]
        avg_prob = np.mean(max_probs)
    print(avg_prob, acc)
    return probs

#== Main plotting function ====================================================================================#

def entropy_plot(probs:dict, labels:dict, ax1, ax2, color='blue', calibrate=False):
    """ get effective number of options distribution """
    probs = convert_dict_np(probs)
    labels = convert_dict_np(labels)
    preds = probs_to_pred(probs)
    hits = get_hits(preds, labels)

    if calibrate: probs = anneal_probs(probs, labels)
    entropies = calc_entropy(probs)
    eff_num_opt = 2**entropies

    # get effective number of options for correctly answered questions
    probs_c = [prob for prob, hit in zip(probs, hits) if hit==1] 
    entropies_c = calc_entropy(probs_c)
    eff_num_opt_c = 2**entropies_c
    
    # binning points and calculating accuracies 
    bins = [i/100 for i in range(100,401, 20)]
    hist, bin_edges = np.histogram(eff_num_opt, bins=bins)
    hist_c, _ = np.histogram(eff_num_opt_c, bins=bins)

    # plotting
    bin_centres = np.array(bins[:-1]) + 0.1
    accuracies  = np.array(hist_c)/np.array(hist)

    ax1.plot(bin_centres, hist, marker='.', color=color, linewidth=4, markersize=18)
    ax2.plot(bin_centres, accuracies, marker='.', linestyle=(0, (3, 1, 1, 1)), color=color, linewidth=4, markersize=18)

def mut_info_plot(probs:dict, probs_sh:dict, labels:dict, ax1, ax2, color='blue', calibrate=False):
    """ rank points by mutual information, and look at accuracy for each bin """
    probs    = convert_dict_np(probs)
    probs_sh = convert_dict_np(probs_sh)
    labels   = convert_dict_np(labels)

    if calibrate: 
        probs    = anneal_probs(probs,    labels)  
        probs_sh = anneal_probs(probs_sh, labels)  

    #get predictions and hits for both models
    preds  = probs_to_pred(probs)
    hits   = get_hits(preds, labels)
    preds_sh = probs_to_pred(probs_sh)
    hits_sh = get_hits(preds_sh, labels)
    
    # get mutual information for all and correct samples
    mut_info      = calc_mutual_information(probs, probs_sh)
    mut_info_c    = [m for m, hit in zip(mut_info, hits) if hit==1] 
    mut_info_c_sh = [m for m, hit in zip(mut_info, hits_sh) if hit==1] 

    # binning points and calculating accuracies 
    hist, bin_edges = np.histogram(mut_info)
    hist_c, _    = np.histogram(mut_info_c, bins=bin_edges)
    hist_c_sh, _ = np.histogram(mut_info_c_sh, bins=bin_edges)

    # plotting
    bin_centres   = np.array(bin_edges[:-1]) + (bin_edges[1]-bin_edges[0])/2
    accuracies    = np.array(hist_c)/np.array(hist)
    accuracies_sh = np.array(hist_c_sh)/np.array(hist)

    ax1.plot(bin_centres, accuracies,  color='orange', linewidth=4, marker='.', markersize=18, label='Q+{O}+C')
    ax1.plot(bin_centres, accuracies_sh,  color='blue', linewidth=4, marker='.', markersize=18, label='Q+{O}')
    ax2.plot(bin_centres, hist, marker='.', color='green', linewidth=4, linestyle=(0, (3, 1, 1, 1)), markersize=18, label='Count')
