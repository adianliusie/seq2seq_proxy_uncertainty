import matplotlib.pyplot as plt
import numpy as np

from scipy import special, stats
from sklearn.metrics import precision_recall_curve

#== Performance Metric Utils ======================================================#
def calc_acc(preds:dict=None, labels:dict=None, probs:dict=None):
    if probs is not None:
        preds = {idx:np.argmax(prob) for prob in probs}
    else:
        assert preds is not None, "probs and preds can't both be provided"

    assert preds.keys() == labels.keys(), "keys don't match"
    hits = sum([preds[idx] == labels[idx] for idx in labels.keys()])
    acc = hits/len(preds)
    return 100*acc

def calc_Fscore(probs:list, labels:list, k=1):
    P, R, thresholds = precision_recall_curve(labels, probs)
    return _calc_Fscore(P, R, k)

def _calc_Fscore(P:list, R:list, k=1):
    def f_score(p, r):
        return (1+k**2)*(p*r)/(p*k**2 + r)
    
    F = [(f_score(p, r), p, r) for p, r in zip(P, R)]
    return max(F)

def plot_best_point(P, R, k, **kwargs):
    f_best, p_best, r_best = _calc_Fscore(P, R, k)
    plt.scatter(r_best, p_best, s=50, **kwargs)

def plot_PR_curve(probs, labels, k=1, **kwargs):
    P, R, thresholds = precision_recall_curve(labels, probs)
    p = plt.plot(R[:-1], P[:-1], **kwargs)
    plot_best_point(P, R, k)
    plt.axhline(y=P[0], linestyle='--', color=p[0].get_color(), alpha=0.3)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('R')
    plt.ylabel('P')

#== Calibrations Utils ===========================================================#
def anneal_probs(probs:dict, labels:dict, silent=True):
    assert probs.keys() == labels.keys()
    
    keys = labels.keys()
    probs_np  = np.array([probs[idx]  for idx in keys])
    labels_np = np.array([labels[idx] for idx in keys])

    #get current model probs and avg max prob 
    logits = np.log(probs_np)  
    max_probs = [max(i) for i in probs_np]
    avg_prob = np.mean(max_probs)

    #look at current model accuracy
    preds_np = np.argmax(probs_np, axis=-1)
    hits = sum(preds_np == labels_np)/len(labels)
    acc = np.mean(hits)
    if not silent: print(f'PRE: {100*avg_prob:.2f} {100*acc:.2f}')

    #do the annealing
    a = 1
    while avg_prob > acc:  
        a += 0.001 if len(keys) < 5000 else 0.005
        annealed_logits = logits/a
        probs  = special.softmax(annealed_logits, axis=1)
        max_probs = [max(i) for i in probs]
        avg_prob = np.mean(max_probs)
                
    if not silent: print(f'CAL: {100*avg_prob:.2f} {100*acc:.2f}')
    
    #recast to dictionary
    probs = {idx:probs[k] for k, idx in enumerate(keys)}
    return probs

