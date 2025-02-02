import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import umap
import numpy as np

import plotly.express as px
from PIL import Image
import io

from sklearn.metrics import confusion_matrix, jaccard_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

import torch

colors = [
    "#FF0000",   # Red
    "#00FF00",   # Green
    "#0000FF",   # Blue
    "#FF00FF",   # Magenta
    "#00FFFF",   # Cyan
    "#800000",   # Maroon
    "#008000",   # Green (dark)
    "#000080",   # Navy
    "#808000",   # Olive
    "#800080",   # Purple
    "#008080",   # Teal
    "#FF8000",   # Orange
    "#FF0080",   # Pink
    "#80FF00",   # Lime
    "#0080FF",   # Sky Blue
    "#8000FF",   # Indigo
    "#FF8080",   # Peach
    "#80FF80",   # Light Green
    "#8080FF",   # Periwinkle
    "#FFFF80",   # Pale Yellow
    "#FF80FF",   # Lavender
    "#80FFFF",   # Light Cyan
    "#C00000",   # Dark Red
    "#00C000",   # Medium Green
    "#0000C0",   # Dark Blue
    "#C0C000",   # Dark Yellow
    "#C000C0",   # Dark Magenta
    "#00C0C0",   # Dark Cyan
    "#C08000",   # Brown
    "#C00080",   # Dark Pink
    "#80C000",   # Olive Green
    "#0080C0",   # Medium Blue
    "#8000C0",   # Violet
    "#C08080",   # Mauve
    "#80C080",   # Medium Light Green
    "#8080C0",   # Light Blue
    "#C0C080",   # Light Yellow
    "#C080C0",   # Light Magenta
    "#80C0C0",   # Light Cyan
    "#E08000",   # Burnt Orange
    "#E00080",   # Raspberry
    "#80E000",   # Lime Green
    "#0080E0",   # Cornflower Blue
    "#8000E0",   # Electric Purple
    "#E08080",   # Salmon
    "#FFFF00",   # Yellow
]

def get_colors(n=5):
    return colors[:n]


import colorsys

def generate_colors(hex_color, n):
    # Convert the hex color to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    
    # Convert RGB to HSL (Hue, Saturation, Lightness)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    # Convert RGB to HSV (Hue, Saturation, Value)
    #h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

    # Calculate the step size for lightness to maximize contrast
    step = 1 / (n + 3) # add 2 extra, to avoid using the extremes
    
    # Generate n contrasting colors by adjusting lightness
    contrasting_colors = []
    for i in range(1, n + 1):
        new_l = min(1.0, max(0.0, step + i * step))
        new_s = min(1.0, max(0.0, step + i * step))
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, new_s)

        #new_v = min(1.0, max(0.0, i * step))
        #new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)

        # Convert RGB to hex format
        new_hex_color = "#{:02X}{:02X}{:02X}".format(
            int(new_r * 255), int(new_g * 255), int(new_b * 255))
        
        contrasting_colors.append(new_hex_color)
    
    return contrasting_colors


def plot_stats(mode:str, history, save_plot=False, filename='stats'):
    if mode == 'clustering':
        plt.figure(figsize=(16, 20))
        rows, cols = 4, 3
    else:
        plt.figure(figsize=(16, 16)) #figsize=(8, 6), dpi=150
        rows, cols = 3, 3
    
    if mode == 'standard':
        plt.suptitle('CMA: standard mode')
    elif mode == 'conditional':
        plt.suptitle('CMA: conditional mode')
    elif mode == 'clustering':
        plt.suptitle('CMA: clustering mode')

    epoch_hist = [x['epoch'] for x in history]
    n_modalities = len(history[0]['loss_recon_modal'])

    #

    plt.subplot(rows, cols, 1)

    loss_recon_modal = [x['loss_recon_modal'] for x in history]
    for modal in zip(*loss_recon_modal):
        plt.plot(epoch_hist, modal)

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss_recon_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})
    
    #
    
    plt.subplot(rows, cols, 2)
    
    loss_kl_modal = [x['loss_kl_modal'] for x in history]
    for modal in zip(*loss_kl_modal):
        plt.plot(epoch_hist, modal)

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss_kl_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})
    
    #
    
    plt.subplot(rows, cols, 3)
    
    loss_d_modal = [x['loss_d_modal'] for x in history]
    loss_d_train_modal = [x['loss_d_train_modal'] for x in history]
    for modal in zip(*loss_d_modal):
        plt.plot(epoch_hist, modal)
    for modal in zip(*loss_d_train_modal):
        plt.plot(epoch_hist, modal)

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    leg = ['loss_d_' + str(i+1) for i in range(n_modalities)]
    leg += ['loss_d_train' + str(i+1) for i in range(n_modalities)] 
    plt.legend(leg, prop={'size': 8})

    #
    
    adjust = 1
    if mode == 'conditional' or mode == 'clustering':
        plt.subplot(rows, cols, 4)

        loss_cond_modal = [x['loss_cond_modal'] for x in history]
        for modal in zip(*loss_cond_modal):
            plt.plot(epoch_hist, modal)

        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss_cond_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})
    
        adjust -= 1

    #

    if mode == 'clustering':
        plt.subplot(rows, cols, 5-adjust)

        loss_clust_modal = [x['loss_clust_modal'] for x in history]
        for modal in zip(*loss_clust_modal):
            plt.plot(epoch_hist, modal)

        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss_clust_' + str(i+1) for i in range(n_modalities)], prop={'size': 8}, loc='upper right')
    
        adjust -= 1
    adjust += 1
    
    #
    
    loss_d = [x['loss_d'] for x in history]
    loss_d_train = [x['loss_d_train'] for x in history]
    
    plt.subplot(rows, cols, 6-adjust)
    plt.plot(epoch_hist, loss_d)
    plt.plot(epoch_hist, loss_d_train)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    legend = ['loss_d', 'loss_d_train']
    plt.legend(legend, prop={'size': 8})
    
    #
    
    loss_recon = [x['loss_recon'] for x in history]
    loss_kl = [x['loss_kl'] for x in history]
    if mode == 'conditional' or mode =='clustering':
        loss_cond = [x['loss_cond'] for x in history]
    if mode =='clustering':
        loss_clustering = [x['loss_clust'] for x in history]

    plt.subplot(rows, cols, 7-adjust)
    plt.plot(epoch_hist, loss_recon)
    plt.plot(epoch_hist, loss_kl)
    if mode == 'conditional' or mode =='clustering':
        plt.plot(epoch_hist, loss_cond)
    if mode =='clustering':
        plt.plot(epoch_hist, loss_clustering)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    legend = ['loss_recon', 'loss_kl', 'loss_cond', 'loss_clust']
    plt.legend(legend, prop={'size': 8})

    #
    
    loss = [x.get('loss') for x in history]
    
    plt.subplot(rows, cols, 8-adjust)
    plt.plot(epoch_hist, loss)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss'], prop={'size': 8})

    #
    
    plt.subplot(rows, cols, 9-adjust)

    acc_d_real_modal = [x['acc_d_real_modal'] for x in history]
    for modal in zip(*acc_d_real_modal):
        plt.plot(epoch_hist, modal)

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc_d_real_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})

    #

    plt.subplot(rows, cols, 10-adjust)

    acc_d_fake_modal = [x['acc_d_fake_modal'] for x in history]
    for modal in zip(*acc_d_fake_modal):
        plt.plot(epoch_hist, modal)

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc_d_fake_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})

    #

    if mode =='clustering':
        plt.subplot(rows, cols, 11-adjust)

        acc_clust_modal = [x['acc_clust_modal'] for x in history]
        for modal in zip(*acc_clust_modal):
            plt.plot(epoch_hist, modal)

        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['acc_clust_' + str(i+1) for i in range(n_modalities)], prop={'size': 8})
    
    #

    plt.tight_layout()

    if save_plot:
        plt.savefig(filename+'.png')
    else:
        plt.show()


def run_pca(data, type='2d', standardize:bool=False):
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data = data_scaled

    if type == '2d':
        pca = PCA(n_components=2)
    elif type == '3d':
        pca = PCA(n_components=3)
    else:
        assert("pca: type must be 2d or 3d")
        
    data_pca = pca.fit_transform(data)

    explained_variance = pca.explained_variance_ratio_

    return data_pca, explained_variance

def run_umap(data, type='2d'):
    if type == '2d':
        reducer = umap.UMAP(n_neighbors=10, n_components=2, init='random', random_state=42)
    elif type == '3d':
        reducer = umap.UMAP(n_neighbors=10, n_components=3, init='random', random_state=42)
    else:
        assert("umap: type must be 2d or 3d")
        
    transformed_data = reducer.fit_transform(data)

    return transformed_data

def plot_scatter_2d(data, labels:list, colors:list, title, save_plot=False, filename='plot'):
    labels = np.array(labels)
    colors = np.array(colors)
    unique_handles = []
    unique_labels = []
    for label in np.unique(labels):
        mask = labels == label
        scatter = plt.scatter(data[mask, 0], data[mask, 1], c=colors[mask], label=labels[mask], s=14)
        #scatter = plt.scatter(data[mask, 0], data[mask, 1], c=colors[mask], label=label, s=14)
        unique_handles.append(scatter)
        unique_labels.append(label)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(title)
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(handles=unique_handles, labels=unique_labels, loc='upper left', bbox_to_anchor=(1, 1))

    if save_plot:
        # Set figure size
        #fig = plt.gcf()
        #fig.set_size_inches(10, 6)
        plt.savefig(filename+'.png', bbox_inches='tight', dpi=100) # to avoid cutting off part of the legend
    else:
        plt.show()


def plot_scatter_3d(data, labels, colors, title, as_gif=False, filename='plot_3d'):
    color_map = {l:c for c,l in set(zip(colors, labels))}

    if not as_gif:
        fig_3d = px.scatter_3d(
            data, x=0, y=1, z=2,
            color = labels,
            color_discrete_map = color_map,
            labels={'color': 'modality'}
        )
        fig_3d.update_traces(
            marker=dict(size=1.5),
            textfont=dict(size=12)
        )
        fig_3d.update_layout(
            title={
                'text': title,
                'y':0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=800,
            height=800,
            legend=dict(
                title=dict(text="Clusters", font=dict(size=16)),  # Legend title font size
                font=dict(size=14),     # Legend item font size
                itemsizing='constant'   # Make the legend marker size constant
            ),
            autosize=False
        )
        
        fig_3d.show()
    else:
        frames = []

        rotation_speed = 1.25

        print('Generating frame:', end=' ')
        
        for angle in np.arange(0, 360, rotation_speed):
            print('.', end='')
            
            # Update the camera orientation for each frame
            #camera['eye']['x'] = 1.25 * np.cos(np.radians(angle))
            #camera['eye']['y'] = 1.25 * np.sin(np.radians(angle))
            eye_x = 1.25 * np.cos(np.radians(angle))
            eye_y = 1.25 * np.sin(np.radians(angle))
        
            fig_3d = px.scatter_3d(
                data, x=0, y=1, z=2,
                color = labels,
                color_discrete_map= color_map,
                labels={'color': 'modality'}
            )
            fig_3d.update_traces(
                marker=dict(size=2.5),
                textfont=dict(size=12)
            )
            fig_3d.update_layout(
                title={
                    'text': title,
                    'y':0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                width=600,
                height=600,
                legend=dict(
                    title=dict(text="Clusters", font=dict(size=16)),  # Legend title font size
                    font=dict(size=14),     # Legend item font size
                    itemsizing='constant'   # Make the legend marker size constant
                ),
                #autosize=False,
                scene=dict(
                    aspectmode='manual',
                    xaxis_title='X-axis',
                    yaxis_title='Y-axis',
                    zaxis_title='Z-axis',
                    camera = dict(
                        eye=dict(x=eye_x, y=eye_y, z=1.25),  # Initial camera position and zoom level
                        center=dict(x=0, y=0, z=0),  # Center point of the plot
                        up=dict(x=1, y=0, z=1),  # Up vector for the camera
                    )
                )
            )
        
            #fig_3d.show()
        
            # Render the frame and append it to the frames list
            img_bytes = fig_3d.to_image(format="png")
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(img)
        
            if angle >= 60: break
        
        # Create a GIF from the frames
        frames[0].save(
            filename + '.gif',
            save_all=True,
            append_images=frames[1:],
            duration=40,  # Adjust the duration (in milliseconds) between frames
            loop=0,  # Set loop to 0 for infinite loop, or specify the number of loops
        )
        
        print(f"\nGIF saved as {filename}")

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


def kmeans_clustering(data, n_clusters, true_labels):
    from sklearn.cluster import KMeans
    from scipy.cluster.vq import kmeans, vq
    from sklearn.preprocessing import normalize
        
    # Regular Kmeans
    #kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    #kmeans.fit(data)
    #pred_labels = kmeans.labels_
    #pred_labels_str = [str(l) for l in pred_labels]
    
    # Spherical Kmeans
    normalized_data = normalize(data)
    centroids, _ = kmeans(normalized_data, n_clusters)
    kmeans_pred_labels, _ = vq(normalized_data, centroids)

    # Remapping
    kmeans_acc, kmeans_cm, remapped_kmeans_labels = remap_confusion_matrix(true_labels, kmeans_pred_labels)

    return kmeans_acc, kmeans_cm, remapped_kmeans_labels

def score_vector(target_labels, pred_labels):
    ji = jaccard_score(target_labels, pred_labels, average='macro')
    print(f"Jaccard Index (JI): {ji}")
    
    ari = adjusted_rand_score(target_labels, pred_labels)
    print(f"Adjusted Rand Index (ARI): {ari}")

'''
def joined_batches_from_dataloader(dataloader):
    all_batches_X = []
    all_batches_y = []
    for batch in dataloader:
        X, y = batch
        all_batches_X.append(X)
        all_batches_y.append(y)
    conc_X = torch.cat(all_batches_X, dim=0)
    conc_y = torch.cat(all_batches_y, dim=0)
    return conc_X, conc_y
'''

