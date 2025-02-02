import numpy as np
import torch
from sklearn.metrics import confusion_matrix, jaccard_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

class Log:

    def __init__(self) -> None:
        self.batch = dict()
        self.history = list()
    
    def log_batch(self, stats:dict):
        for k in stats:
            if not k in self.batch:
                self.batch[k] = list()
            self.batch[k].append(stats[k])

    def log_epoch(self):
        epoch_stats = dict()
        for k in self.batch:
            if not k in epoch_stats:
                epoch_stats[k] = list()
            if '_modal' in k:
                res = []
                for m in zip(*self.batch[k]):
                    res.append(sum(m) / len(m))
                epoch_stats[k] = res
            else:
                epoch_stats[k] = sum(self.batch[k]) / len(self.batch[k])
            self.batch[k] = list()
        self.history.append(epoch_stats)

    def get_history(self):
        return self.history
    
    def get_last_epoch(self):
        return self.history[-1]

# Some functions

def modal_target(target:int, size:int):
    targets = torch.full((size,), target).long()
    return targets

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / output.size(0)



def remap_confusion_matrix(labels:list, pred_labels:list):
    '''Compute confusion matrix based on best assignment'''
    cm = confusion_matrix(labels, pred_labels)
    def _make_cost_m(cm):
        s = np.max(cm)
        return (-cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    remap_cm = cm[:, indexes[1]]
    acc = np.trace(remap_cm) / np.sum(remap_cm)

    original_indexes, replacement_indexes = indexes
    #label_mapping = dict(zip(original_indexes, replacement_indexes))
    #remapped_labels = np.vectorize(label_mapping.get)(pred_labels)
    
    #names = sorted(set(labels + pred_labels))
    names = sorted(np.unique(np.concatenate((labels, pred_labels))))
    remapped_names = [names[ri] for ri in replacement_indexes]
    label_mapping = dict(zip(remapped_names, names))

    #remapped_labels = [label_mapping[l] for l in pred_labels]
    remapped_labels = np.vectorize(label_mapping.get)(pred_labels)
    
    return acc, remap_cm, remapped_labels

def score_vector(target_labels, pred_labels):
    ji = jaccard_score(target_labels, pred_labels, average='macro')
    print(f"Jaccard Index (JI): {ji}")
    
    ari = adjusted_rand_score(target_labels, pred_labels)
    print(f"Adjusted Rand Index (ARI): {ari}")
