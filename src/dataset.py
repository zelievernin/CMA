import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

root_dir = os.path.join(os.path.dirname(__file__), '../data')

class IntersimDataset(Dataset):
    """Load a expression dataset generated with interSIM"""

    def __init__(self, data, labels, transform=None):
        self.df = torch.tensor(data.values, dtype=torch.float32)
        self.labels = labels
        self.transform = transform
        self.classes = torch.unique(torch.tensor(labels.loc[:,'cluster.id'].values)) - 1 # adjust to zero index

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df[idx,:]
        y = self.labels.loc[self.labels.index[idx],'cluster.id'] - 1 # adjust to zero index
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def get_paired_data(suffix):
    expr_X = pd.read_csv(root_dir + '/paired/expression_' + suffix + '.txt', sep='\t')
    methyl_X = pd.read_csv(root_dir + '/paired/methylation_' + suffix + '.txt', sep='\t')
    protein_X = pd.read_csv(root_dir + '/paired/protein_' + suffix + '.txt', sep='\t')

    y_common = pd.read_csv(root_dir + '/paired/clusters_' + suffix + '.txt', sep='\t', index_col=0)

    return (expr_X, y_common), (methyl_X, y_common), (protein_X, y_common)


def get_unpaired_data(suffix):
    expr_X = pd.read_csv(root_dir + '/unpaired/expression_unpaired_' + suffix + '.txt', sep='\t')
    expr_y = pd.read_csv(root_dir + '/unpaired/clusters_expression_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    methyl_X = pd.read_csv(root_dir + '/unpaired/methylation_unpaired_' + suffix + '.txt', sep='\t')
    methyl_y = pd.read_csv(root_dir + '/unpaired/clusters_methylation_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    protein_X = pd.read_csv(root_dir + '/unpaired/protein_unpaired_' + suffix + '.txt', sep='\t')
    protein_y = pd.read_csv(root_dir + '/unpaired/clusters_protein_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    return (expr_X, expr_y), (methyl_X, methyl_y), (protein_X, protein_y)


def generate_datasets(suffix='5_diff', type='unpaired', train=True, test=False):
    if type == 'paired':
        expr, methyl, protein = get_paired_data(suffix)
        print('Loading paired dataset')
    elif type == 'unpaired':
        expr, methyl, protein = get_unpaired_data(suffix)
        print('Loading unpaired dataset')
    else:
        raise Exception('Invalid dataset type')

    expr_X, expr_y = expr
    methyl_X, methyl_y = methyl
    protein_X, protein_y = protein

    expr_train_X, expr_test_X, expr_train_y, expr_test_y = train_test_split(expr_X, expr_y, test_size=0.2, random_state=53)
    exp_train_dataset = IntersimDataset(expr_train_X, expr_train_y)
    exp_test_dataset = IntersimDataset(expr_test_X, expr_test_y)

    methyl_train_X, methyl_test_X, methyl_train_y, methyl_test_y = train_test_split(methyl_X, methyl_y, test_size=0.2, random_state=53)
    methyl_train_dataset = IntersimDataset(methyl_train_X, methyl_train_y)
    methyl_test_dataset = IntersimDataset(methyl_test_X, methyl_test_y)

    protein_train_X, protein_test_X, protein_train_y, protein_test_y = train_test_split(protein_X, protein_y, test_size=0.2, random_state=53)
    protein_train_dataset = IntersimDataset(protein_train_X, protein_train_y)
    protein_test_dataset = IntersimDataset(protein_test_X, protein_test_y)

    datasets = list()
    if train:
        datasets.extend([exp_train_dataset, methyl_train_dataset, protein_train_dataset])
    if test:
        datasets.extend([exp_test_dataset, methyl_test_dataset, protein_test_dataset])
    
    return datasets
