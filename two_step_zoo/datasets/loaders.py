import numpy as np
import torch
from torch.utils.data import DataLoader

from .image import get_image_datasets
from .generated import get_generated_datasets
from .calo_challenge_showers import get_physics_datasets
from .calo_challenge_epl import get_physics_datasets_epl
from .supervised_dataset import SupervisedDataset


def get_loaders_from_config(cfg):
    """
    Wrapper function providing frequently-used functionality.

    Updates `cfg` with dataset information.
    """
    train_loader, valid_loader, test_loader = get_loaders(
        cfg,
        dataset=cfg["dataset"],
        data_root=cfg.get("data_root", "data/"),
        make_valid_loader=cfg["make_valid_loader"],
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg["valid_batch_size"],
        test_batch_size=cfg["test_batch_size"]
    )

    train_dataset = train_loader.dataset.x
    cfg["train_dataset_size"] = train_dataset.shape[0]
    cfg["data_shape"] = tuple(train_dataset.shape[1:])
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))
    train_labels = train_loader.dataset.y
    cfg["label_dim"] = int(train_labels.shape[-1])

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader

def get_loaders(
        cfg,
        dataset,
        data_root,
        make_valid_loader,
        train_batch_size,
        valid_batch_size,
        test_batch_size
):
    if dataset in ["celeba", "mnist", "fashion-mnist", "cifar10", "svhn"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, make_valid_loader)

    elif dataset in ["sphere", "klein", "two_moons"]:
        train_dset, valid_dset, test_dset = get_generated_datasets(dataset)

    elif dataset in ["photons1", "pions1", "electrons2", "electrons3"]:
        train_dset, valid_dset, test_dset = get_physics_datasets(cfg, dataset, data_root, make_valid_loader)
        
    elif dataset in ["photons1-epl", "pions1-epl", "electrons2-epl", "electrons3-epl"]:
        train_dset, valid_dset, test_dset = get_physics_datasets_epl(cfg, dataset, data_root, make_valid_loader)

    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_loader = get_loader(train_dset, train_batch_size, drop_last=True, pin_memory=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, valid_batch_size, drop_last=False, pin_memory=True)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, test_batch_size, drop_last=False, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_loader(dset, batch_size, drop_last, shuffle=True, **loader_kwargs):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )


def get_embedding_loader(embeddings, batch_size, drop_last, role, y=None):
    dataset = SupervisedDataset(
        name="embeddings",
        role=role,
        x=embeddings,
        y=y
    )
    return get_loader(dataset, batch_size, drop_last)


def get_eperlayer_loader(energies, batch_size, drop_last, role, x=None):
    if x is None:
        x = torch.zeros(energies.shape[0]) # dummy data to satisfy SupervisedDataset assert statement, not used.
    dataset = SupervisedDataset(
        name="sampled_eperlayers",
        role=role,
        x=x,
        y=energies
    )
    return get_loader(dataset, batch_size, drop_last)


def remove_drop_last(loader):
    dset = loader.dataset
    return get_loader(dset, loader.batch_size, False)
