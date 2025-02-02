config = {
    'mode': 'clustering',
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-4,
    'alpha': 1,
    'beta': 1,
    'gamma': 1,
    'lambda': 1e-3,
    'xi': 1,
    'latent_dims': 10,
    'vae': [
        {
            'n_inputs': 131,
            'n_hidden': 256
        },
        {
            'n_inputs': 367,
            'n_hidden': 256
        },
        {
            'n_inputs': 160,
            'n_hidden': 256
        }
    ],
    'discriminator': {
        'n_hidden': 256
    },
    'clustering_module': {
        'n_hidden': 100
    },
    'checkpoint_prefix': 'checkpoint',
    'saving_freq': None,
    'device': 'cuda'
}
