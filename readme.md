# Cross-modal autoencoder (CMA) with clustering module


## Conda environment

 Generate a conda enviroment to run the model.

 ```bash
conda create --name cma python=3.11 -c conda-forge
conda activate cma
pip install Jupyterlab matplotlib numpy pandas seaborn scikit-learn torch umap-learn
```

## Datasets

Two version of a datasets were generated with interSIM, a paired and an unpaired one.
Each has the same 3 modalities (expression, methylation, protein) and there are versions with 5, 10, and 15 clusters.
In addition, each cluster has two version, one where all the clusters have different number of elements and another were 
some of the clusters have equal number of elements.

The function generate_datasets from the datasets.py file returns thee datasets for training, testing or both.

example:
```python
datasets = generate_datasets(suffix='5_diff', type='unpaired', train=True, test=False)
# Generate the dataloaders
dataloaders = [DataLoader(ds, batch_size=32, drop_last=True, shuffle=False) for ds in test_datasets]
```


## Training

For training just run the training script.

```python
python3 train.py
```

The script reads the configuration from `config.py` given as a python dict. The provided configuration has the optimal parameters
for the 5_diff unpaired dataset. If you wish to replace just one or multiple of the parameters in the config file you can do
it from the command line.

```python
python3 train.py --epochs 10
```

Run `python3 train.py --help` for a full description.

The trained model is store in the `./checkpoint` folde.


## Training report

The provided jupiter notebook `training_report.py` loads the specified checkpoint and allows to visualize the training statistics and latent space. To run type:

## Predict

The `predict.py` script returns the prediction from the conditional classifier of the clustering module depending on the mode that the module was trained on.

## Issues

When displaying the confusion matrix in the `training_report.py` jupyter notebook, sometimes only the first row of the confusion matrix appears while the rest is blank. This is a known issue of the seaborn.heatmap function. If this happens try installing a different version.