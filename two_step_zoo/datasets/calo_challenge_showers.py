import h5py
from pathlib import Path
import numpy as np
import torch
import os

from two_step_zoo.datasets.supervised_dataset import SupervisedDataset
from two_step_zoo.datasets.calo_challenge_epl import en_binning, get_eperlayer, preprocess_eperlayer, preprocess_einc, undo_preprocess_einc, undo_preprocess_eperlayer
from two_step_zoo.evaluators.xml_handler import XMLHandler


ALPHA = 1e-6 # for numerical stability of log

# NOTE: Data loading and preprocessing modifies code from https://github.com/ViniciusMikuni/CaloScore
def preprocess(shower, energy, cfg):
    dataset = cfg["dataset"]
    particle = {'photons1': 'photon', 'pions1': 'pion',
                    'electrons2': 'electron', 'electrons3': 'electron'}[dataset]
    xml_filename = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), "evaluators", "binning", f"binning_dataset_{dataset}.xml")
    xml = XMLHandler(particle, filename=xml_filename)
    relevantLayers = xml.GetRelevantLayers()
    eperlayer = None
    # normalize so that the voxels sum up to one for each layer
    if cfg["normalize_to_epl"]:
        if cfg.get("conditional_on_epl", False) is False:
            raise Exception("EPL conditioning is needed to undo normalization to energies per layer.")
        bin_edges = xml.GetBinEdges()
        if eperlayer is None:
                eperlayer = get_eperlayer(dataset, shower)
        prep_shower = np.empty_like(shower)
        for idx,l in enumerate(relevantLayers):
            prep_shower[:, bin_edges[l]:bin_edges[l+1]] = np.ma.divide(shower[:, bin_edges[l]:bin_edges[l+1]], eperlayer[:,idx].reshape(-1,1)).filled(0) 
    else:
        # rescale to GeV
        prep_shower = shower / 1000.0
        energy = energy / 1000.0
        # Rescale by incident energy times a dataset-dependent constant
        # to ensure that voxel energies are in [0,1]
        prep_shower = prep_shower / (cfg["max_deposited_ratio"] * energy)

    if cfg["normalized_deposited_energy"]:
        if cfg["normalize_to_epl"]:
            raise Exception("Normalization wrt deposited energy is not supposed to be used together with normalization wrt layer energies.")
        if "electrons" in dataset:
            raise ValueError(f"Warning: 'normalized_deposited_energy' only meant to be used with flat data, not {dataset}")
        # Compute total energy deposited in each shower and append as a feature to the input x
        deposited_energy = np.sum(prep_shower, -1, keepdims=True)
        prep_shower = np.concatenate((prep_shower, deposited_energy), -1)

        if cfg["deposited_energy_per_layer"]:
            # Append eperlayer as features to the input x
            if eperlayer is None:
                eperlayer = get_eperlayer(dataset, prep_shower)
            prep_shower = np.concatenate((prep_shower, eperlayer), -1)
        
        # Normalize voxel data by deposited energy (some showers deposit zero energy)
        raw_shape = cfg["raw_shape"][1]
        prep_shower[:, :raw_shape] = np.ma.divide(prep_shower[:, :raw_shape], deposited_energy).filled(0)
    
    if cfg.get("conditional_on_epl", False):
        # For three step models, append eperlayer (computed on non-prepr showers) to the conditioning label y
        if eperlayer is None:
            eperlayer = get_eperlayer(dataset, shower)
        prep_eperlayer = preprocess_eperlayer(eperlayer, energy, cfg)
        prep_einc = preprocess_einc(energy, cfg)
        energy = np.concatenate((prep_einc, prep_eperlayer), -1)
    elif cfg["logspace_incident_energies"]:
        energy = np.log10(energy / cfg["energy_min"]) / np.log10(cfg["energy_max"] / cfg["energy_min"])
    else:
        energy = (energy - cfg["energy_min"]) / (cfg["energy_max"] - cfg["energy_min"])

    # Convert to image-like if needed
    if "electrons" in dataset:
        prep_shower = prep_shower.reshape(cfg["raw_shape"])

    # Transform voxel energies to logit space
    if cfg["logitspace_voxels"]:
        x = ALPHA + (1 - 2*ALPHA) * prep_shower
        prep_shower = np.ma.log(x / (1 - x)).filled(0)

    return prep_shower, energy


def undo_preprocessing_showers(shower, energy, cfg):
    '''Revert the transformations applied to the training set'''

    dataset = cfg["dataset"]

    particle = {'photons1': 'photon', 'pions1': 'pion',
                    'electrons2': 'electron', 'electrons3': 'electron'}[dataset]
    xml_filename = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), "evaluators", "binning", f"binning_dataset_{dataset}.xml")

    xml = XMLHandler(particle, filename=xml_filename)
    relevantLayers = xml.GetRelevantLayers()

    eperlayer = None


    if cfg["logitspace_voxels"]:
        # stable sigmoid
        x = np.where(shower < 0, np.exp(shower)/(1 + np.exp(shower)), 1/(1 + np.exp(-shower)))
        shower = (x - ALPHA) / (1 - 2*ALPHA)

    # Flatten if needed
    if "electrons" in dataset:
        shower = shower.reshape((shower.shape[0],-1))

    if cfg.get("conditional_on_epl", False):
        # Keep dim by slicing for single element
        einc = energy[:, :1]
        eperlayer = energy[:, 1:]

        inc_energy = undo_preprocess_einc(einc,cfg)
        eperlayer = undo_preprocess_eperlayer(eperlayer,inc_energy,cfg)
    elif cfg["logspace_incident_energies"]:
        inc_energy = cfg['energy_min'] * (cfg['energy_max'] / cfg['energy_min'])**energy
    else:
        inc_energy = cfg['energy_min'] + (cfg['energy_max'] - cfg['energy_min']) * energy
    if cfg['normalized_deposited_energy']:
        if cfg["normalize_to_epl"]:
            raise Exception("Normalization wrt deposited energy is not supposed to be used together with normalization wrt layer energies")
        # Keep dim by slicing for single element
        energies = shower[:, cfg['raw_shape'][1]:cfg['raw_shape'][1]+1]
        shower = shower[:, :cfg['raw_shape'][1]]
        shower = shower * energies

    if cfg["normalize_to_epl"]:   
        bin_edges = xml.GetBinEdges()
        postp_shower = np.empty_like(shower)
        for idx,l in enumerate(relevantLayers):
            postp_shower[:, bin_edges[l]:bin_edges[l+1]] = shower[:, bin_edges[l]:bin_edges[l+1]]*eperlayer[:,idx].reshape(-1,1)

    else:
        # clip to fix numerical imprecision, or samples with negative values
        shower = np.clip(shower, 0.0, 1.0)
        # change shape for broadcasting
        shape = [1]*shower.ndim
        shape[0] -= 2
        shower = shower * cfg['max_deposited_ratio'] * inc_energy.reshape(shape)

        # rescale to MeV
        postp_shower = shower * 1000.0
        inc_energy = inc_energy * 1000.0

    if "photons" in cfg["dataset"] or "pions" in cfg["dataset"]:
        inc_energy = en_binning(inc_energy)
    
    return postp_shower, inc_energy

def get_raw_data_arrays(cfg: dict, name: str, root: str, split: str):
    # Train and test filenames.
    filenames = {
        "photons1": [['dataset_1_photons_1.hdf5'], ['dataset_1_photons_2.hdf5']],
        "pions1": [['dataset_1_pions_1.hdf5'], ['dataset_1_pions_2.hdf5']],
        "electrons2": [['dataset_2_1.hdf5'], ['dataset_2_2.hdf5']],
        "electrons3": [['dataset_3_1.hdf5', 'dataset_3_2.hdf5'], ['dataset_3_3.hdf5', 'dataset_3_4.hdf5']],
        }
    data_path = lambda x: Path(root) / x
    showers = []
    energies = []
    if split == "train":
        datasets = filenames[name][0]
    elif split == "test":
        datasets = filenames[name][1]
    for dataset in datasets:
        with h5py.File(data_path(dataset), "r") as h5f:
            energy = h5f['incident_energies'][:].astype(np.float32)
            shower = h5f['showers'][:].astype(np.float32)
            if cfg["preprocess_physics_data"]:
                shower, energy = preprocess(shower, energy, cfg)
            showers.append(shower)
            energies.append(energy)
    if showers:
        shape = list(showers[0].shape)
        shape[0] = -1
        showers = np.reshape(showers, shape)
        energy_shape = list(energies[0].shape)
        energy_shape[0] = -1
        energies = np.reshape(energies, energy_shape)

    return showers, energies

def get_physics_datasets(cfg, name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.2 if make_valid_dset else 0

    if not name in ["photons1", "pions1", "electrons2", "electrons3"]:
        raise ValueError(f"Unknown dataset {name}")

    showers_train, energies_train = get_raw_data_arrays(cfg, name, data_root, "train")
    showers_test, energies_test = get_raw_data_arrays(cfg, name, data_root, "test")

    perm = torch.randperm(showers_train.shape[0])
    shuffled_showers = showers_train[perm]
    shuffled_energies = energies_train[perm]

    valid_size = int(valid_fraction * showers_train.shape[0])
    valid_showers = torch.tensor(shuffled_showers[:valid_size], dtype=torch.get_default_dtype())
    valid_energies = torch.tensor(shuffled_energies[:valid_size], dtype=torch.get_default_dtype())
    train_showers = torch.tensor(shuffled_showers[valid_size:], dtype=torch.get_default_dtype())
    train_energies = torch.tensor(shuffled_energies[valid_size:], dtype=torch.get_default_dtype())

    train_dset = SupervisedDataset(name, "train", train_showers, train_energies)
    valid_dset = SupervisedDataset(name, "valid", valid_showers, valid_energies)

    test_showers = torch.tensor(showers_test, dtype=torch.get_default_dtype())
    test_energies = torch.tensor(energies_test, dtype=torch.get_default_dtype())
    test_dset = SupervisedDataset(name, "test", test_showers, test_energies)

    return train_dset, valid_dset, test_dset
