#!/usr/bin/env python3

import argparse
import torch
import time
import h5py
import random
import numpy as np
from pathlib import Path

from config import load_epl_configs
from two_step_zoo import (
    get_single_module, get_two_step_module, get_trainer, get_single_trainer, get_writer
)
from two_step_zoo.datasets import undo_preprocessing_showers, preprocess_einc, get_loader, undo_preprocessing_epl, preprocess_eperlayer
from two_step_zoo.datasets.supervised_dataset import FastDataset
from two_step_zoo.evaluators.evaluate import evaluate_samples


t0 = time.time()

parser = argparse.ArgumentParser(description="Script for Sampling from Three Step Calo Challenge Density Estimator as Fast as Possible")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Attempt to load the best_valid checkpoint first.")
parser.add_argument("--batch-size", type=int, default=100,
    help="Batch size for sampling.")
parser.add_argument("--evaluate", action="store_true",
    help="Evaluate showers after sampling, takes extra time.")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
args.only_test = True
args.load_pretrained_epl = False
args.load_pretrained_gae = False
args.max_epochs_loaded_epl = False
args.max_epochs_loaded_gae = False
args.max_epochs_loaded_de = False
args.max_epochs_loaded = False

 # Set random seeds for reproducibility
np.random.seed(seed=args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

epl_cfg, gae_cfg, de_cfg, shared_cfg = load_epl_configs(args=args)
epl_cfg['test_batch_size'] = args.batch_size
shared_cfg['test_batch_size'] = args.batch_size
args.dataset = shared_cfg['dataset']

writer = get_writer(args, write_cfg=False, epl_cfg=epl_cfg, gae_cfg=gae_cfg, de_cfg=de_cfg, shared_cfg=shared_cfg)

# EPL training routine
filenames = {
        "photons1": ['dataset_1_photons_2.hdf5'],
        "pions1": ['dataset_1_pions_2.hdf5'],
        "electrons2": ['dataset_2_2.hdf5'],
        "electrons3": ['dataset_3_3.hdf5', 'dataset_3_4.hdf5'],
        }
if shared_cfg['dataset'] == "electrons3":
    print("electrons3 not handled in fast sampling")
data_path = lambda x: Path(shared_cfg.get("data_root", "data/")) / x

datasets = filenames[shared_cfg['dataset']]
for dataset in datasets:
    with h5py.File(data_path(dataset), "r") as h5f:
        raw_energy = h5f['incident_energies'][:].astype(np.float32)
        if epl_cfg["preprocess_physics_data"]:
            prep_einc = preprocess_einc(raw_energy, epl_cfg)

epl_energies = torch.tensor(prep_einc, dtype=torch.get_default_dtype()).squeeze(dim=0)
shared_energies = torch.tensor(raw_energy, dtype=torch.get_default_dtype()).squeeze(dim=0)
epl_dset = FastDataset(epl_energies)
shared_dset = FastDataset(shared_energies)
epl_loader = get_loader(epl_dset, shared_cfg['test_batch_size'], drop_last=False, shuffle=False, pin_memory=True)
shared_loader = get_loader(shared_dset, shared_cfg['test_batch_size'], drop_last=False, shuffle=False, pin_memory=True)

data_dims = {"photons1": 5, "pions1": 7, "electrons2": 45, "electrons3": 45}
epl_cfg["label_dim"] = 1
epl_cfg["data_dim"] = data_dims[shared_cfg['dataset']]
epl_cfg["data_shape"] = tuple([epl_cfg["data_dim"]])

# EPL networks
epl_module = get_single_module(
    epl_cfg,
    data_dim=epl_cfg["data_dim"],
    data_shape=epl_cfg["data_shape"],
    label_dim=epl_cfg["label_dim"],
    train_dataset_size=epl_energies.shape[0]
).to(device)
# Trainer loads pre-trained weights
epl_trainer = get_single_trainer(
    module=epl_module,
    ckpt_prefix="epl",
    writer=writer,
    cfg=epl_cfg,
    train_loader=None,
    valid_loader=None,
    test_loader=epl_loader,
    evaluator=None,
    only_test=args.only_test,
)
checkpoint_load_list = ["latest", "best_valid"]
if args.load_best_valid_first: checkpoint_load_list = checkpoint_load_list[::-1]
for ckpt in checkpoint_load_list:
    try:
        epl_trainer.load_checkpoint(ckpt)
        break
    except FileNotFoundError:
        print(f"Did not find {ckpt} epl checkpoint")

# Two-step networks
shower_data_dims = {"photons1": 368, "pions1": 533, "electrons2": 6480, "electrons3": 40500}
shared_cfg["data_dim"] = shower_data_dims[shared_cfg['dataset']]
shared_cfg["data_shape"] = tuple([shared_cfg["data_dim"]])
shared_cfg["label_dim"] = 1+data_dims[shared_cfg['dataset']]
shared_cfg["train_dataset_size"] = epl_energies.shape[0]
two_step_module = get_two_step_module(gae_cfg, de_cfg, shared_cfg).to(device)
# Trainer loads pre-trained weights
trainer = get_trainer(
    two_step_module=two_step_module,
    writer=writer,
    gae_cfg=gae_cfg,
    de_cfg=de_cfg,
    shared_cfg=shared_cfg,
    train_loader=None,
    valid_loader=None,
    test_loader=shared_loader,
    gae_evaluator=None,
    de_evaluator=None,
    shared_evaluator=None,
    load_best_valid_first=args.load_best_valid_first,
    pretrained_gae_path=args.load_dir if args.load_pretrained_gae else "",
    freeze_pretrained_gae=None,
    only_test=args.only_test
)

# Sampling routine
with torch.no_grad():
    if not trainer.gae.use_labels:
        print("Cannot create conditional samples when labels are not provided")
        exit()
    num_samples = len(epl_loader.dataset)

    tt0 = time.time()
    samples_created = 0
    conditional_energies = None
    dataloader_iterator = enumerate(epl_loader)
    shared_iterator = iter(shared_loader)
    while samples_created < num_samples:
        try:
            j, y = next(dataloader_iterator)
            raw_e_inc = next(shared_iterator)

            if samples_created + y.shape[0] > num_samples:
                y = y[:num_samples - samples_created]
            samples_created += y.shape[0]

            epl_e_inc = y.to(epl_module.device)
            eperlayer_batch = epl_module.sample(epl_e_inc.shape[0], epl_e_inc)
            # undo preprocess of the epl model and preprocess according to the two_step model
            eperlayer, _ = undo_preprocessing_epl(eperlayer_batch, epl_e_inc, epl_cfg)
            raw_e_inc = raw_e_inc.to(epl_module.device)
            if shared_cfg["normalize_to_epl"]:
                pass
            else:
                raw_e_inc = raw_e_inc / 1000.0
            if shared_cfg.get("conditional_on_epl", False):
                prep_eperlayer = preprocess_eperlayer(eperlayer, raw_e_inc, shared_cfg)
                prep_einc = preprocess_einc(raw_e_inc, shared_cfg)
                energy = torch.cat((prep_einc, prep_eperlayer), dim=-1)
            elif shared_cfg["logspace_incident_energies"]:
                energy = np.log10(energy / shared_cfg["energy_min"]) / np.log10(shared_cfg["energy_max"] / shared_cfg["energy_min"])
            else:
                energy = (energy - shared_cfg["energy_min"]) / (shared_cfg["energy_max"] - shared_cfg["energy_min"])

            try:
                shower_batch = two_step_module.sample(energy.shape[0], energy).cpu().numpy()
            except AttributeError:
                print("No sample method available")

            energy = energy.cpu().numpy()
            if conditional_energies is None:
                conditional_energies = energy
                showers = shower_batch
            else:
                conditional_energies = np.concatenate((conditional_energies, energy), axis=0)
                showers = np.concatenate((showers, shower_batch), axis=0)
        except (StopIteration):
            continue

    # Sampling can generate values out of expected range for undo_preprocessing
    showers = np.clip(showers, two_step_module.data_min.cpu().numpy(), two_step_module.data_max.cpu().numpy())
    showers, conditional_energies = undo_preprocessing_showers(showers, conditional_energies, shared_cfg)
    showers = showers.reshape(showers.shape[0], -1) # flatten
    tt1 = time.time()

    print(f"{num_samples} conditional samples generated in {tt1 - tt0} seconds")
    data_dict = {
        'incident_energies': conditional_energies,
        'showers': showers,
    }
    trainer.writer.write_hdf5('generated_showers', data_dict)

if args.evaluate:
    evaluate_samples(showers, conditional_energies, shared_cfg, writer.logdir)

t1 = time.time()
print(f"CaloMan fast sampling script total time: {t1 - t0} seconds")
