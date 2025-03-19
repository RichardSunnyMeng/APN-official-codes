import os
import torch
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from scipy.special import softmax
from sklearn import metrics


def eval_base(results, binary=True, thr=0.5):
    if "p" in results.keys() and len(results["p"]) > 0:
        try:
            y = torch.cat(results['p'], dim=0)
            gt = torch.cat(results['label'], dim=0)
        except:
            y = np.concatenate(results['p'], axis=0)
            gt = np.concatenate(results['label'], axis=0)
    else:
        try:
            logits = torch.cat(results['logits'], dim=0)
            gt = torch.cat(results['label'], dim=0)
            if len(logits.shape) == 2:
                y = F.softmax(logits, dim=-1)
            else:
                y = F.sigmoid(logits)
            y = y.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
        except:
            logits = np.concatenate(results['logits'], axis=0)
            gt = np.concatenate(results['label'], axis=0)
            if len(logits.shape) == 2:
                y = softmax(logits, axis=-1)
            else:
                y = 1 / (1 + np.exp(-logits))
    if binary:
        gt = gt
        if len(y.shape) == 2:
            y = y[:, 1]

        fpr, tpr, _ = metrics.roc_curve(gt, y)
        y_clone = deepcopy(y)
        
        y[y > thr] = 1
        y[y < thr] = 0
        acc = metrics.accuracy_score(gt, y.astype(np.int))
        recall = metrics.recall_score(gt, y.astype(np.int))
        precision = metrics.precision_score(gt, y.astype(np.int))

        y[y_clone < thr] = 1
        y[y_clone > thr] = 0
        gt = np.abs(gt - 1)
        recall_rev = metrics.recall_score(gt, y.astype(np.int))
        precision_rev = metrics.precision_score(gt, y.astype(np.int))
        return {'auc': metrics.auc(fpr, tpr), 'acc': acc, 
                "pos_recall": recall, "pos_precision": precision,
                "neg_recall": recall_rev, "neg_precision": precision_rev,
                }
    else:
        raise NotImplementedError

def eval_MVFA(results, binary=True, thr=0.5):

    B = results['label'][0].shape[0]
    N = results['logits'][0].shape[0]

    p = results['logits']
    p_new = []
    for sub_results in p:
        layers = np.split(sub_results, N // B, 0)
        sub_results_new = layers[0]
        for layer in layers[1:]:
            sub_results_new = sub_results_new + layer
        sub_results_new = sub_results_new / len(layers)
        p_new.append(sub_results_new)


    y = np.concatenate(p_new, axis=0)
    gt = np.concatenate(results['label'], axis=0)


    if binary:
        gt = gt
        if len(y.shape) == 2:
            y = y[:, 1]

        fpr, tpr, _ = metrics.roc_curve(gt, y)
        y_clone = deepcopy(y)
        
        y[y > thr] = 1
        y[y < thr] = 0
        acc = metrics.accuracy_score(gt, y.astype(np.int))
        recall = metrics.recall_score(gt, y.astype(np.int))
        precision = metrics.precision_score(gt, y.astype(np.int))

        y[y_clone < thr] = 1
        y[y_clone > thr] = 0
        gt = np.abs(gt - 1)
        recall_rev = metrics.recall_score(gt, y.astype(np.int))
        precision_rev = metrics.precision_score(gt, y.astype(np.int))
        return {'auc': metrics.auc(fpr, tpr), 'acc': acc, 
                "pos_recall": recall, "pos_precision": precision,
                "neg_recall": recall_rev, "neg_precision": precision_rev,
                }
    else:
        raise NotImplementedError