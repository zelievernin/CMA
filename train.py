import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config as default_config
from src.dataset import generate_datasets
from src.models import CMA
from src.losses import ClusteringLoss
from src.functions import Log, modal_target, accuracy, remap_confusion_matrix

script_dir = os.path.dirname(__file__)


class Trainer:

    def __init__(self, model):
        self.model = model
        self.optimizers = dict()      
        self.loss_fn = dict()
        self.log = Log()

    def train_clustering_module(self, multi_modal_data:list):
        # Outputs
        output_latents, output_recon = self.model.forward_vae(multi_modal_data, train=True, encoder_only=False)
        output_projector, output_cluster = self.model.forward_clustering_module(output_latents, train=True, apply_argmax=False)

        # Losses

        loss_recon = []
        for i in range(len(output_recon)):
            l = self.loss_fn['recon'](multi_modal_data[i][0], output_recon[i])
            loss_recon.append(l)
        _loss_recon = self.model.config['alpha'] * torch.stack(loss_recon).sum()

        loss_kl = []
        for net in self.model.vae:
            loss_kl.append(net.encoder.kl)
        _loss_kl = self.model.config['lambda'] * torch.stack(loss_kl).sum()

        loss_clustering = []
        for i in range(len(output_projector)):
            l = self.loss_fn['clustering'](output_projector[i], output_cluster[i], output_latents[i])
            loss_clustering.append(l)
        _loss_clustering = self.model.config['xi'] * torch.stack(loss_clustering).sum()

        loss = _loss_recon + _loss_kl + _loss_clustering

        # Backpropagate and step

        loss.backward()
        
        self.optimizers['vae'].step()
        self.optimizers['clustering'].step()

        # Accuracy of predictions

        acc = list()
        for i in range(len(output_cluster)):
            pred = torch.argmax(torch.nn.functional.softmax(output_cluster[i], dim=1), axis=1)
            current_acc, _, _ = remap_confusion_matrix(multi_modal_data[i][1].cpu(), pred.cpu())
            acc.append(current_acc)

        # Statistics
        stats = {
            'loss_clust': _loss_clustering.item(),
            'loss_clust_modal': [],
            'acc_clust_modal': []
        }
        for i in range(len(multi_modal_data)):
            stats['loss_clust_modal'].append(loss_clustering[i].item())
            stats['acc_clust_modal'].append(acc[i].item())
        self.log.log_batch(stats)

    def train_discriminator(self, multi_modal_data:list):
        # Outputs
        output_latents = self.model.forward_vae(multi_modal_data, train=True, encoder_only=True)
        output_d = self.model.forward_d(output_latents, train=True)
       
        # Compute the loss
        
        loss_d_train = list()
        for i in range(len(output_d)):
            loss_d_train.append(self.loss_fn['clf'](output_d[i], modal_target(i, output_d[i].size(0)).to(self.model.config['device'])))
        _loss_d_train = self.model.config['gamma'] * torch.stack(loss_d_train).sum()


        # Backpropagate and step

        _loss_d_train.backward()
        self.optimizers['d'].step()

        # Statistics
        stats = {
            'loss_d_train': _loss_d_train.item(),
            'loss_d_train_modal': [],
            'acc_d_real_modal': [],
            'acc_d_fake_modal': []
        }
        for i in range(len(multi_modal_data)):
            stats['loss_d_train_modal'].append(loss_d_train[i].item())
            stats['acc_d_real_modal'].append(accuracy(output_d[i], modal_target(i, multi_modal_data[i][0].size(0)).to(self.model.config['device'])))

            acc = 0
            for i2 in range(len(multi_modal_data)):
                if i == i2: continue
                target = modal_target(i2, multi_modal_data[i2][0].size(0)).to(self.model.config['device'])
                acc += accuracy(output_d[i], target)
            stats['acc_d_fake_modal'].append(acc / (len(multi_modal_data) - 1))

        self.log.log_batch(stats)

    def train_vae_and_cond_classif(self, multi_modal_data:list):
        # Outputs

        output_latents, output_recon = self.model.forward_vae(multi_modal_data, train=True, encoder_only=False)
        output_d = self.model.forward_d(output_latents, train=False)

        if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
            output_cond = self.model.forward_conditional_classifier(output_latents)

        if self.model.config['mode'] == 'clustering':
            _, pred_clusters = self.model.forward_clustering_module(output_latents, train=True, apply_argmax=True)


        # Losses

        loss_recon = []
        for i in range(len(output_recon)):
            loss = self.loss_fn['recon'](multi_modal_data[i][0], output_recon[i])
            loss_recon.append(loss)
        _loss_recon = self.model.config['alpha'] * torch.stack(loss_recon).sum()

        loss_kl = []
        for net in self.model.vae:
            loss_kl.append(net.encoder.kl)
        _loss_kl = self.model.config['lambda'] * torch.stack(loss_kl).sum()

        targets = list()
        for i in range(len(output_d)):
            targets.append(modal_target(i, output_d[i].size(0)).to(self.model.config['device']))
        loss_d = list()
        for i in range(len(output_d)):
            l = 0
            for i2 in range(len(output_d)):
                if i == i2: continue
                l += self.loss_fn['clf'](output_d[i], targets[i2])
            l *= 1 / (len(output_d)-1)
            loss_d.append(l)
        _loss_d = self.model.config['gamma'] * torch.stack(loss_d).sum()

        if self.model.config['mode'] == 'conditional':
            loss_cond = list()
            for i in range(len(output_cond)):
                loss_cond.append(self.loss_fn['clf'](output_cond[i], multi_modal_data[i][1]))
        elif self.model.config['mode'] == 'clustering':
            loss_cond = list()
            for i in range(len(pred_clusters)):
                loss_cond.append(self.loss_fn['clf'](output_cond[i], pred_clusters[i]))

        loss = _loss_recon + _loss_kl + _loss_d

        if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
            _loss_cond = self.model.config['beta'] * torch.stack(loss_cond).sum()
            loss += _loss_cond

        # Backpropagate and step
        
        loss.backward()

        self.optimizers['vae'].step()
        if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
            self.optimizers['cond'].step()

        # Statistics

        stats = {
            'loss_recon': _loss_recon.item(),
            'loss_recon_modal': [],
            'loss_kl': _loss_kl.item(),
            'loss_kl_modal': [],
            'loss_d': _loss_d.item(),
            'loss_d_modal': [],
            'loss': loss.item()
        }
        for i in range(len(multi_modal_data)):
            stats['loss_recon_modal'].append(loss_recon[i].item())
            stats['loss_kl_modal'].append(loss_kl[i].item())
            stats['loss_d_modal'].append(loss_d[i].item())

        if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
            stats['loss_cond_modal'] = []
            for i in range(len(multi_modal_data)):
                stats['loss_cond_modal'].append(loss_cond[i].item())
            stats['loss_cond'] = _loss_cond.item()

        self.log.log_batch(stats)

    def run(self, datasets, n_classes=None):
        self.model.set_model(n_classes)

        print(f"Training in {self.model.config['mode']} mode")

        # Set up the optimizers
        self.optimizers['vae'] = torch.optim.Adam([{'params': net.parameters()} for net in self.model.vae], lr=self.model.config['lr'])
        self.optimizers['d'] = torch.optim.Adam(self.model.d.parameters(), lr=self.model.config['lr'])
        if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':      
            self.optimizers['cond'] = torch.optim.Adam(self.model.cond.parameters(), lr=self.model.config['lr'])
        if self.model.config['mode'] == 'clustering':
            self.optimizers['clustering'] = torch.optim.Adam(
                [
                    {'params': self.model.projector.parameters()},
                    {'params': self.model.cluster.parameters()}
                ],
                lr=self.model.config['lr']
            )

        # Set loss functions
        self.loss_fn['recon'] = nn.MSELoss()
        self.loss_fn['clf'] = nn.CrossEntropyLoss()
        if self.model.config['mode'] == 'clustering':
            self.loss_fn['clustering'] = ClusteringLoss(n_classes)

        #
        dataloaders = [DataLoader(ds, batch_size=self.model.config['batch_size'], drop_last=True, shuffle=True) for ds in datasets]

        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + self.model.config['epochs']):
            self.log.log_batch({'epoch': epoch}) 
            for multi_modal_data in zip(*dataloaders):
                multi_modal_data = [(X.to(self.model.config['device']), y.to(self.model.config['device'])) for (X, y) in multi_modal_data]
                #print(epoch, multi_modal_data[0][0].shape, multi_modal_data[1][0].shape, multi_modal_data[2][0].shape)

                # Train clustering module
                if self.model.config['mode'] == 'clustering':
                    self.train_clustering_module(multi_modal_data)
                
                # Train discriminator
                self.train_discriminator(multi_modal_data)

                # Train VAEs and conditional classifier
                self.train_vae_and_cond_classif(multi_modal_data)

            #print('---')
            self.log.log_epoch()

            if self.model.config['saving_freq'] == None or epoch % self.model.config['saving_freq'] == 0:
                state = {
                    'config': self.model.config, #vars(args),
                    'models_names': {
                        'vae': [net.__class__.__name__ for net in self.model.vae],
                        'd': self.model.d.__class__.__name__
                    },
                    'models_states': [
                        [net.state_dict() for net in self.model.vae],
                        self.model.d.state_dict()
                    ],
                    'optimizers_states': [self.optimizers[opt].state_dict() for opt in self.optimizers],
                    'history': self.log.get_history()
                }

                if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
                    state['models_names']['cond'] = self.model.cond.__class__.__name__
                    state['models_states'].append(self.model.cond.state_dict())

                if self.model.config['mode'] == 'clustering':
                    state['models_names']['projector'] = self.model.projector.__class__.__name__
                    state['models_names']['cluster'] = self.model.cluster.__class__.__name__
                    state['models_states'].append(self.model.projector.state_dict())
                    state['models_states'].append(self.model.cluster.state_dict())

                if not os.path.isdir(script_dir + '/checkpoint'):
                    os.mkdir(script_dir + '/checkpoint')

                if self.model.config['saving_freq'] == 0:
                    torch.save(state, script_dir + '/checkpoint/' + self.model.config['checkpoint_prefix'] + '.pth')
                else:
                    torch.save(state, script_dir + '/checkpoint/' + self.model.config['checkpoint_prefix'] + '_' + str(epoch) + '.pth')

            # Display log

            log = self.log.get_last_epoch()
            display_log = f"epoch={int(log['epoch'])} loss_recon={log['loss_recon']:.5f} loss_kl={log['loss_kl']:.5f} loss_d={log['loss_d']:.5f} loss_d_train={log['loss_d_train']:.5f}"
            if self.model.config['mode'] == 'conditional' or self.model.config['mode'] == 'clustering':
                display_log += f" loss_cond={log['loss_cond']:.5f}"
            if self.model.config['mode'] == 'clustering':
                display_log += f" loss_clust={log['loss_clust']:.5f}"
            d_acc = sum(log['acc_d_real_modal']) / len(log['acc_d_real_modal'])
            display_log += f" loss={log['loss']:.5f} avg_d_acc={d_acc:.2f}"
            if self.model.config['mode'] == 'clustering':
                clustering_acc = sum(log['acc_clust_modal']) / len(log['acc_clust_modal'])
                display_log += f" avg_clust_acc={clustering_acc:.2f}"
            print(display_log)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch autoencoder training')
    parser.add_argument('--mode', choices=['standard', 'conditional', 'clustering'], required=False, help='Training mode')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='train for this number of epochs')
    parser.add_argument('--lr', type=float, help='autoencoders and classifiers learning rate')
    parser.add_argument('--alpha', action="store", type=float, help='reconstruction hyperparameter')
    parser.add_argument('--beta', action="store", type=float, help='conditional hyperparameter')
    parser.add_argument('--gamma', action="store", type=float, help='adversarial hyperparameter')
    parser.add_argument('--lambda', action="store", type=float, help='kl hyperparameter')
    parser.add_argument('--xi', action="store", type=float, help='clustering hyperparameter')
    parser.add_argument('--latent-dims', default=10, type=int, help='number of latent dimentions for the VAEs')
    parser.add_argument('--checkpoint-prefix', action='store', help='prefix name for checkpoint file')
    #parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--saving-freq', type=int, help='Save intermediate states of training every this many epochs')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='If not specified cuda device is used if available')
    
    return vars(parser.parse_args())


def train(cma, datasets:list, n_classes:int):
        trainer = Trainer(cma.model)
        trainer.run(datasets, n_classes)

if __name__ == '__main__':
    # Read input arguments, they will overwrite config file settings
    args = parse_arguments()

    if args['device'] == None:
        args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Override config with params
    for k in args:
        if args[k] == None: continue
        default_config[k] = args[k]
    
    if default_config['saving_freq'] == None:
        default_config['saving_freq'] = default_config['epochs'] # only save last epoch

    # Load datasets
    datasets = generate_datasets(suffix='5_diff', type='unpaired', train=True)
    expr_train_dataset, methyl_train_dataset, protein_train_dataset = datasets

    classes = {len(ds.classes) for ds in datasets}
    assert len(classes) == 1, "Uneven number of clases in datasets"
    n_classes = classes.pop()

    cma = CMA(default_config)

    train(
        cma,
        [
            expr_train_dataset, 
            methyl_train_dataset, 
            protein_train_dataset
        ],
        n_classes
    )

    if default_config['mode'] == 'clustering':
        print('\n\nTrain data')
        dataloaders = [DataLoader(ds, batch_size=default_config['batch_size'], drop_last=False, shuffle=False) for ds in datasets]
        cma.get_clustering_module_accuracy(dataloaders)

        print(f'\n\nTest data')
        test_datasets = generate_datasets(suffix='5_diff', type='unpaired', train=False, test=True)
        test_dataloaders = [DataLoader(ds, batch_size=default_config['batch_size'], drop_last=False, shuffle=False) for ds in test_datasets]
        cma.get_clustering_module_accuracy(test_dataloaders)
