import os

import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
        )

        self.z_mean = nn.Linear(n_hidden, latent_dims)
        self.z_log_var = nn.Linear(n_hidden, latent_dims)

        self.kl = 0

    def forward(self, x):       
        out = self.encoder(x)
        mu = self.z_mean(out)
        logvar = self.z_log_var(out)
        
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Reparametrization        
        std = torch.exp(0.5 * logvar)            
        eps = torch.randn_like(std)
        z = mu + eps * std
    
        return z


class Decoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_input),
        )

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    
    def __init__(self, n_input=131, n_hidden=5000, latent_dims=20):
        super().__init__()
        self.encoder = Encoder(n_input, latent_dims, n_hidden)
        self.decoder = Decoder(n_input, latent_dims, n_hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---

class Discriminator(nn.Module):
    
    def __init__(self, nz=128, n_hidden=1024, n_out=1):
        super().__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        return self.net(x)


class CondClassifier(nn.Module):
    
    def __init__(self, nz=128, n_out=1):
        super().__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_out)
        )

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    
    def __init__(self, latents=30, n_out=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latents, n_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Cluster(nn.Module):
    
    def __init__(self, n_in=100, n_clusters=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_in, n_clusters),
            #nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        #out = nn.functional.softmax(out, dim=1)
        return out

# ---

class Model:

    def __init__(self, config):
        self.config = config

        self.vae = list()
        self.d = None
        self.cond = None
        self.projector = None
        self.cluster = None

    def set_model(self, n_classes=None):
        for settings in self.config['vae']:
            v = VAE(n_input=settings['n_inputs'], n_hidden=settings['n_hidden'], latent_dims=self.config['latent_dims']).to(self.config['device'])
            self.vae.append(v)

        self.d = Discriminator(nz=self.config['latent_dims'], n_hidden=self.config['discriminator']['n_hidden'], n_out=len(self.vae)).to(self.config['device'])

        if self.config['mode'] == 'conditional' or self.config['mode'] == 'clustering':
            assert n_classes != None, "The number of classes must be defined" 
            self.cond = CondClassifier(nz=self.config['latent_dims'], n_out=n_classes).to(self.config['device'])

        if self.config['mode'] == 'clustering':
            self.projector = Projector(latents=self.config['latent_dims'], n_out=self.config['clustering_module']['n_hidden']).to(self.config['device'])
            self.cluster = Cluster(n_in=self.config['clustering_module']['n_hidden'], n_clusters=n_classes).to(self.config['device'])

    def forward(self, multi_modal_data:list):
        if self.config['mode'] == 'conditional' or self.config['mode'] == 'clustering':
            output_latents = self.forward_vae(multi_modal_data, train=False, encoder_only=True)

            if self.config['mode'] == 'conditional':    
                output_cond = self.forward_conditional_classifier(output_latents, train=False)
                pred = [torch.argmax(torch.nn.functional.softmax(output_cond[i], dim=1), axis=1).cpu().numpy().tolist() for i in range(len(output_cond))]
            elif self.config['mode'] == 'clustering':
                _, pred = self.forward_clustering_module(output_latents, train=False, apply_argmax=True)
                pred = [m.cpu().numpy().tolist() for m in pred]
            return pred

    def forward_vae(self, multi_modal_data:list, train=False, encoder_only=False):
        for i in range(len(self.vae)):
            if train:
                self.vae[i].train()
                self.vae[i].zero_grad()
            else:
                self.vae[i].eval()
    
        output_latents = list()
        output_recon = list()
        for i, net in enumerate(self.vae):
            X, y = multi_modal_data[i]
            output_latents.append(net.encoder(X))
            if not encoder_only:
                output_recon.append(net.decoder(output_latents[-1]))

        if encoder_only:
            return output_latents
        return output_latents, output_recon

    def forward_clustering_module(self, latents:list, train=False, apply_argmax=False):
        if train:
            self.projector.train()
            self.projector.zero_grad()
            
            self.cluster.train()
            self.cluster.zero_grad()
        else:
            self.projector.eval()
            self.cluster.eval()

        output_projector = list()
        output_cluster = list()
        for current_latent in latents:
            output_projector.append(self.projector(current_latent))
            out = self.cluster(output_projector[-1])
            if apply_argmax:
                out = torch.argmax(nn.functional.softmax(out, dim=1), axis=1)
            output_cluster.append(out)
        
        return output_projector, output_cluster

    def forward_d(self, latents:list, train=False):
        if train:
            self.d.train()
            self.d.zero_grad()
        else:
            self.d.eval()

        output_d = list()
        for current_latent in latents:
            output_d.append(self.d(current_latent))

        return output_d

    def forward_conditional_classifier(self, latents:list, train=False):
        if train:
            self.cond.train()
            self.cond.zero_grad()
        else:
            self.cond.eval()

        output_cond = list()
        for current_latent in latents:         
            output_cond.append(self.cond(current_latent))
        
        return output_cond


from src.functions import remap_confusion_matrix, score_vector

class CMA:

    def __init__(self, config=None) -> None:
        self.model = Model(config)

    def load(self, checkpoint_file:str, device):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        checkpoint['device'] = device
        self.model.config = checkpoint['config']

        n_classes = None
        if len(checkpoint['models_states']) >= 3 and checkpoint['models_states'][2] != None:
            n_classes = checkpoint['models_states'][2]['net.0.weight'].size(0)

        self.model.set_model(n_classes)

        for i in range(len(self.model.vae)):
            self.model.vae[i].load_state_dict(checkpoint['models_states'][0][i])

        self.model.d.load_state_dict(checkpoint['models_states'][1])

        if self.model.cond != None:
            self.model.cond.load_state_dict(checkpoint['models_states'][2])

        if self.model.projector != None and self.model.cluster != None:
            self.model.projector.load_state_dict(checkpoint['models_states'][3])
            self.model.cluster.load_state_dict(checkpoint['models_states'][4])

        print('==> Pre-trained model loaded')

    def get_clustering_module_accuracy(self, dataloaders: list):
        if self.model.projector != None and self.model.cluster != None:

            def get_joined_latents_and_labels(dataloader, vae_model):
                latents = []
                labels = []
                pred_labels = []

                vae_model.eval()
                self.model.projector.eval()
                self.model.cluster.eval()
                for X, y in dataloader:
                    with torch.no_grad():
                        vae_latents = vae_model.encoder(X.to(self.model.config['device']))
                        pred_clusters = torch.argmax(nn.functional.softmax(self.model.cluster(self.model.projector(vae_latents)), dim=1), axis=1)
                
                    latents.append(vae_latents)
                    labels.append(y)
                    pred_labels.append(pred_clusters)
                
                conc_latents = torch.cat(latents, dim=0)
                conc_labels = torch.cat(labels, dim=0).cpu()
                conc_pred_labels = torch.cat(pred_labels, dim=0).cpu()

                #
                acc, conf_mat, remapped_labels = remap_confusion_matrix(conc_labels, conc_pred_labels)
                
                return conc_latents, conc_labels, remapped_labels, acc, conf_mat # vae_latents, true_labels, pred_labels, pred_acc, confusion_matrix


            joined_data = list()
            for i in range(len(dataloaders)):
                joined_data.append(get_joined_latents_and_labels(dataloaders[i], self.model.vae[i]))

            conc_latents = torch.cat([data[0] for data in joined_data], dim=0)
            conc_labels = torch.cat([data[1] for data in joined_data], dim=0)

            self.model.projector.eval()
            self.model.cluster.eval()
            pred_clusters = torch.argmax(nn.functional.softmax(self.model.cluster(self.model.projector(conc_latents)), dim=1), axis=1)
            conc_acc, conc_conf_mat, conc_remapped_labels = remap_confusion_matrix(conc_labels, pred_clusters.cpu())

            print()
            print('All data')
            print(f"Accuracy: {conc_acc}")
            print(f"Confusion matrix:")
            print(conc_conf_mat)
            score_vector(conc_labels, conc_remapped_labels)

            for i in range(len(joined_data)):
                print()
                print(f'Modality {i} only')
                print(f"Accuracy: {joined_data[i][3]}")
                print(f"Confusion matrix:")
                print(joined_data[i][4])
                score_vector(joined_data[i][1], joined_data[i][2])

    def predict(self, multi_modal_data:list):
        return self.model.forward(multi_modal_data)