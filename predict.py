import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.models import CMA
from src.dataset import generate_datasets


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch autoencoder training')
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=False, help='If not specified cuda device is used if available')
    parser.add_argument('--checkpoint-fn',  type=str, required=True, help='Checkpoint filename')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_arguments()

    device = args['device']
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_file = args['checkpoint_fn']
    assert os.path.exists(checkpoint_file), 'Error: invalid checkpoint file'

    cma = CMA()
    cma.load(checkpoint_file, device)
    
    # Load data
    datasets = generate_datasets(suffix='5_diff', type='unpaired', train=False, test=True)
    dataloaders = [DataLoader(ds, batch_size=32, drop_last=False, shuffle=False) for ds in datasets]

    for multi_modal_data in zip(*dataloaders):
        multi_modal_data = [(X.to(device), y.to(device)) for (X, y) in multi_modal_data]

        pred = cma.predict(multi_modal_data)
        print(pred)
        break