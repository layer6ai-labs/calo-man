import numpy as np

from two_step_zoo.datasets.calo_challenge_showers import get_physics_datasets


def load_data(cfg, train=True):
    dataset = cfg["dataset"].lower()
    if dataset in ['photons1', 'pions1', 'electrons2', 'electrons3']:            
        train_dset, _, test_dset = get_physics_datasets(cfg, cfg["dataset"], cfg["data_root"], cfg["make_valid_loader"])
        if train:
            dset = train_dset
        else:
            dset = test_dset
        dset = dset.x.numpy()
        if dataset == 'pions1': #dedup
            dset = np.unique(dset, axis=0)
        if cfg["max_num_samples"] != -1:
            rng = np.random.default_rng(cfg['seed'])
            dset = rng.choice(dset, size=cfg["max_num_samples"], replace=False)
        shape = dset.shape
        print(f"Using {dataset}, {shape[0]} datapoints, {shape[1]} features.")

    elif dataset == "random-gaussian":
        rng = np.random.default_rng(cfg['seed'])
        dset = rng.standard_normal(cfg["shape"])

        num_samples = cfg["shape"][0]
        dimension = cfg["shape"][1]
        print(f"Generating {num_samples} datapoints of dimension {dimension} normally distributed")

    elif dataset == "random-uniform":
        rng = np.random.default_rng(cfg['seed'])
        dset = rng.uniform(size=cfg["shape"])

        num_samples = cfg["shape"][0]
        dimension = cfg["shape"][1]
        print(f"Generating {num_samples} datapoints of dimension {dimension} uniformly distributed")

    else:
        raise Exception("Dataset not understood")

    return dset
