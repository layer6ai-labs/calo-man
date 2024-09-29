import os
import h5py
from pathlib import Path
import numpy as np
import torch

from two_step_zoo.datasets.supervised_dataset import SupervisedDataset
from two_step_zoo.evaluators.high_level_features import HighLevelFeatures
import two_step_zoo.datasets.loaders as loaders_package


ALPHA = 1e-6 # for numerical stability of log

def en_binning(energies):
    """
    fix numerical errors when returning incident energies for photons and pions by binning
    """
    target_en = [256,512,1024,2048,4096,8192,16384,32768,
                65536,131072,262144,524288,1048576,2097152,4194304]
    bins = [128,384,768,1536,3072,6144,12288,24576,49152,
            98304,196608,393216,786432,1572864,3145728,6291456]

    if isinstance(energies, np.ndarray):
        target_en = np.array(target_en).reshape(-1,1)
        return target_en[np.digitize(energies.flatten(),bins)-1]
    else:
        target_en = torch.tensor(target_en).reshape(-1,1)
        return target_en[torch.bucketize(energies.flatten(),torch.tensor(bins))-1]


def get_eperlayer(dataset, shower):
    # get layer information from xml
    particle = {'photons1': 'photon', 'pions1': 'pion',
                    'electrons2': 'electron', 'electrons3': 'electron'}[dataset]
    xml_filename = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), "evaluators", "binning", f"binning_dataset_{dataset}.xml")
    hlf = HighLevelFeatures(particle, xml_filename)
    layer_end = [hlf.bin_edges[idx] for idx in hlf.relevantLayers]
    layer_end.pop(0)
    layer_end.append(hlf.bin_edges[-1])
    
    # Compute total energy deposited in each layer
    eperlayer = []
    prev_layer_end_idx = 0
    for layer_end_idx in layer_end:
        layer_voxels = shower[:, prev_layer_end_idx:layer_end_idx]
        prev_layer_end_idx = layer_end_idx
        layer_deposited_energy = np.sum(layer_voxels, -1, keepdims=True)
        eperlayer.append(layer_deposited_energy)
    eperlayer = np.concatenate(eperlayer, axis=-1)
    
    return eperlayer

def preprocess_eperlayer(eperlayer, einc, cfg):
    if cfg["caloflow_epl"]:
        ie_factor = cfg["max_deposited_ratio"]
        einc = einc.reshape(-1)
        n_layers = len(eperlayer.T)
        prep_eperlayer = np.zeros_like(eperlayer)
        einc=einc*ie_factor
        the_sum=0
        temp_prepr_epl = np.zeros_like(eperlayer)
        for layer in range(n_layers):
            new_energies=eperlayer[:,layer]
            the_sum=the_sum+new_energies
            temp_prepr_epl[:,layer]=new_energies
        prep_eperlayer[:,0]=(the_sum)/einc
        for k in range(n_layers):
            if k==0:
                continue
            dividend=0
            for d in range(k-1,n_layers):
                dividend=dividend+temp_prepr_epl[:,d]
            prep_eperlayer[:,k]=np.ma.divide(temp_prepr_epl[:,k-1],dividend).filled(0)
        x_0 = ALPHA + (1 - 2 * ALPHA) * prep_eperlayer
        prep_eperlayer = np.ma.log(x_0 / (1 - x_0)).filled(0)

    else:
        if isinstance(eperlayer, np.ndarray):
            prep_eperlayer = np.log10((eperlayer+.001)/1e5) - 1
        else:
            prep_eperlayer = torch.log10((eperlayer+.001)/1e5) - 1
    
    return prep_eperlayer

def preprocess_einc(einc, cfg):
    if cfg["caloflow_epl"]: 
        if isinstance(einc, np.ndarray):
            prep_einc = np.log10(einc/(33.3*1e3))
        else:
            prep_einc = torch.log10(einc/(33.3*1e3))
    else: 
        if isinstance(einc, np.ndarray):
            prep_einc = np.log10((einc+.001))
        else:
            prep_einc = torch.log10((einc+.001))

    return prep_einc

def preprocess(eperlayer, einc, cfg):  
    prep_eperlayer = preprocess_eperlayer(eperlayer, einc, cfg)
    prep_einc = preprocess_einc(einc, cfg)
    return prep_eperlayer, prep_einc

def undo_preprocess_eperlayer(prep_eperlayer, einc, cfg):
    if cfg["caloflow_epl"]:
        einc = einc.flatten()
        ie_factor = cfg["max_deposited_ratio"]
        if isinstance(prep_eperlayer, np.ndarray):
            exp_step=np.exp(prep_eperlayer)
        else:
            exp_step=torch.exp(prep_eperlayer)
        x=(exp_step/(1+exp_step))
        eperlayer1=(x-ALPHA)/(1-2*ALPHA)
        ie=einc*ie_factor
        n_layers = len(prep_eperlayer.T)
        if isinstance(prep_eperlayer, np.ndarray):
            eperlayer=np.zeros_like(prep_eperlayer)
        else:
            eperlayer=torch.zeros_like(prep_eperlayer)
        eperlayer[:,0]=ie*eperlayer1[:,0]*eperlayer1[:,1]
        for k in range(1,n_layers):
            if k<n_layers-1:
                eperlayer[:,k]=ie*eperlayer1[:,0]*eperlayer1[:,k+1]
            else:
                eperlayer[:,k]=ie*eperlayer1[:,0]
            for j in range(1,k+1):
                eperlayer[:,k]=eperlayer[:,k]*(1-eperlayer1[:,j])
    else: 
        eperlayer = 1e5*10**(prep_eperlayer+1) - .001
    if isinstance(einc, np.ndarray):
        eperlayer = np.clip(eperlayer, 0.0, None)
    else: 
        eperlayer = torch.clip(eperlayer, 0.0)
    return eperlayer

def undo_preprocess_einc(prep_einc, cfg):
    if cfg["caloflow_epl"]: 
        einc = (10**prep_einc)*(33.3*1e3)
    else:
        einc = 10**prep_einc - .001
    return einc

def undo_preprocessing_epl(prep_eperlayer, prep_einc, cfg):    
    prep_einc = prep_einc.reshape(-1)
    einc = undo_preprocess_einc(prep_einc,cfg)
    eperlayer = undo_preprocess_eperlayer(prep_eperlayer, einc, cfg)

    return eperlayer, einc


def get_raw_data_arrays_epl(cfg: dict, name: str, root: str, split: str):
    filenames = {
        "photons1-epl": [['dataset_1_photons_1.hdf5'], ['dataset_1_photons_2.hdf5']],
        "pions1-epl": [['dataset_1_pions_1.hdf5'], ['dataset_1_pions_2.hdf5']],
        "electrons2-epl": [['dataset_2_1.hdf5'], ['dataset_2_2.hdf5']],
        "electrons3-epl": [['dataset_3_1.hdf5', 'dataset_3_2.hdf5'], ['dataset_3_3.hdf5', 'dataset_3_4.hdf5']],
        }

    data_path = lambda x: Path(root) / x
    eperlayers = []
    energies = []
    if split == "train":
        datasets = filenames[name][0]
    elif split == "test":
        datasets = filenames[name][1]
    for dataset in datasets:
        with h5py.File(data_path(dataset), "r") as h5f:
            energy = h5f['incident_energies'][:].astype(np.float32)
            shower = h5f['showers'][:].astype(np.float32)
            eperlayer = get_eperlayer(cfg["dataset"][:-4], shower)
            if cfg["preprocess_physics_data"]:
                eperlayer, energy = preprocess(eperlayer, energy, cfg)
            eperlayers.append(eperlayer)
            energies.append(energy)
    if eperlayers: # if empty skip this
        shape = list(eperlayers[0].shape)
        shape[0] = -1
        eperlayers = np.reshape(eperlayers, shape)
        energy_shape = list(energies[0].shape)
        energy_shape[0] = -1
        energies = np.reshape(energies, energy_shape)

    return eperlayers, energies


def get_physics_datasets_epl(cfg, name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.2 if make_valid_dset else 0

    if not name in ["photons1-epl", "pions1-epl", "electrons2-epl", "electrons3-epl"]:
        raise ValueError(f"Unknown dataset {name}")

    eperlayers_train, energies_train = get_raw_data_arrays_epl(cfg, name, data_root, "train")
    eperlayers_test, energies_test = get_raw_data_arrays_epl(cfg, name, data_root, "test")

    perm = torch.randperm(eperlayers_train.shape[0])
    shuffled_eperlayers = eperlayers_train[perm]
    shuffled_energies = energies_train[perm]

    valid_size = int(valid_fraction * eperlayers_train.shape[0])
    valid_eperlayers = torch.tensor(shuffled_eperlayers[:valid_size], dtype=torch.get_default_dtype())
    valid_energies = torch.tensor(shuffled_energies[:valid_size], dtype=torch.get_default_dtype())
    train_eperlayers = torch.tensor(shuffled_eperlayers[valid_size:], dtype=torch.get_default_dtype())
    train_energies = torch.tensor(shuffled_energies[valid_size:], dtype=torch.get_default_dtype())

    train_dset = SupervisedDataset(name, "train", train_eperlayers, train_energies)
    valid_dset = SupervisedDataset(name, "valid", valid_eperlayers, valid_energies)

    test_eperlayers = torch.tensor(eperlayers_test, dtype=torch.get_default_dtype())
    test_energies = torch.tensor(energies_test, dtype=torch.get_default_dtype())
    test_dset = SupervisedDataset(name, "test", test_eperlayers, test_energies)

    return train_dset, valid_dset, test_dset


def epl_sampled_loaders(train_loader, valid_loader, test_loader, epl_module):
    # Create new loaders that have only y data (E_inc, EPL) with EPL sampled from epl_module
    loaders = [train_loader, valid_loader, test_loader]
    epl_sampled_loaders = []

    for loader in loaders:
        if loader == None:
            epl_sampled_loaders.append(None)
            continue

        # NOTE: If we drop last in this step, we will lose training data in embeddings.
        #       However, we cannot simply change loader.drop_last to False after the loader
        #       is initialized. Thus, we create a new dataloader in remove_drop_last.
        loader_drop_last = loader.drop_last
        if loader_drop_last:
            loader = loaders_package.remove_drop_last(loader)

        combined_context = None
        with torch.no_grad():
            for _, y, _ in loader:
                e_inc = y[:, :1].to(epl_module.device) # single element slice to keep dimension
                try:
                    eperlayer_batch = epl_module.sample(y.shape[0], e_inc).cpu().numpy()
                except:
                    raise AttributeError("No sample method available in epl_module")

                e_inc = e_inc.cpu().numpy()
                energies = np.concatenate((e_inc, eperlayer_batch), axis=-1)
                assert energies.shape == y.shape
                if combined_context is None:
                    combined_context = energies
                else:
                    combined_context = np.concatenate((combined_context, energies), axis=0)

        epl_dataloader = loaders_package.get_eperlayer_loader(
            energies=combined_context,
            batch_size=loader.batch_size,
            drop_last=loader_drop_last,
            role=loader.dataset.role,
        )
        epl_sampled_loaders.append(epl_dataloader)
    
    return (*epl_sampled_loaders, )
