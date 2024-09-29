import time

import torch
import numpy as np
from pytorch_fid import fid_score
import prdc
from two_step_zoo.datasets.loaders import get_loader

from .metrics_helpers import InceptionHelper, HistogramHelper, EPLHistogramHelper, ClassifierHelper, sample_showers
from .ood_helpers import ood_acc


def fid(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64,
        cache=None):
    """
    Following Heusel et al. (2017), compute FID from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    inception = InceptionHelper(module, dataloader, gen_samples, gen_batch_size)

    gen_mu, gen_sigma = inception.compute_inception_stats()

    if cache is None:
        gt_mu, gt_sigma = inception.compute_inception_stats(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = inception.get_inception_features(dataloader)
        cache["gt_feats"] = gt_feats
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma
    elif "gt_stats" not in cache:
        gt_feats = cache["gt_feats"]
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma
    else:
        gt_mu, gt_sigma = cache["gt_stats"]

    return fid_score.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)


def precision_recall_density_coverage(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64, nearest_k=5,
        cache=None):
    """
    Following Naaem et al. (2020), compute Precision, Recall, Density, Coverage from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    inception = InceptionHelper(module, dataloader, gen_samples, gen_batch_size)

    gen_feats = inception.get_inception_features()

    if cache is None:
        gt_feats = inception.get_inception_features(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = inception.get_inception_features(dataloader)
        cache["gt_feats"] = gt_feats
    else:
        gt_feats = cache["gt_feats"]

    return prdc.compute_prdc(gt_feats, gen_feats, nearest_k)


def log_likelihood(module, dataloader, cache=None):
    with torch.no_grad():
        return module.log_prob(dataloader, None).mean()  # second input is dummy thanks to @batch_or_dataloader()


def l2_reconstruction_error(module, dataloader, cache=None):
    with torch.no_grad():
        return module.rec_error(dataloader, None).mean()  # second input is dummy thanks to @batch_or_dataloader()


def loss(module, dataloader, cache=None):
    with torch.no_grad():
        return module.loss(dataloader, None).mean()  # second input is dummy thanks to @batch_or_dataloader()


def null_metric(module, dataloader, cache=None):
    return 0


def ave_histogram_difference(module, dataloader, dataset_name, max_deposited_ratio,
        raw_shape, energy_min, energy_max, normalized_deposited_energy,
        deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
        conditional_on_epl, caloflow_epl, normalize_to_epl, epl_module, gen_samples=50000, cache=None):

    histogram = HistogramHelper(module, dataloader, gen_samples, dataset_name,
        max_deposited_ratio, raw_shape, energy_min, energy_max,
        normalized_deposited_energy, deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
        conditional_on_epl,caloflow_epl,normalize_to_epl)

    # NOTE: Counts are normalized
    if "hist_counts" not in cache:
        gt_feature_counts_dict, e_layers_bin_fn = histogram.get_histogram_counts(loader=dataloader, epl_module=epl_module)
        cache["hist_counts"] = gt_feature_counts_dict
        cache["e_layers_bin_fn"] = e_layers_bin_fn
    else:
        gt_feature_counts_dict = cache["hist_counts"]
        e_layers_bin_fn = cache["e_layers_bin_fn"]

    # Will use a dataloader from the module
    sample_feature_counts_dict, _ = histogram.get_histogram_counts(e_layers_bin_fn=e_layers_bin_fn, epl_module=epl_module)
    return histogram.get_chisq_distances(gt_feature_counts_dict, sample_feature_counts_dict)


def ave_epl_histogram_difference(module, dataloader, dataset_name, max_deposited_ratio,
        caloflow_epl, gen_samples=50000, cache=None):

    histogram = EPLHistogramHelper(module, dataloader, gen_samples, dataset_name,
        max_deposited_ratio, caloflow_epl)

    # NOTE: Counts are normalized
    if "hist_counts" not in cache:
        gt_feature_counts_dict, e_layers_bin_fn = histogram.get_histogram_counts(loader=dataloader)
        cache["hist_counts"] = gt_feature_counts_dict
        cache["e_layers_bin_fn"] = e_layers_bin_fn
    else:
        gt_feature_counts_dict = cache["hist_counts"]
        e_layers_bin_fn = cache["e_layers_bin_fn"]

    # Will use a dataloader from the module
    sample_feature_counts_dict, _ = histogram.get_histogram_counts(e_layers_bin_fn=e_layers_bin_fn)
    return histogram.get_chisq_distances(gt_feature_counts_dict, sample_feature_counts_dict)


def shower_classifier_auc(module, dataloader, dataset_name, max_deposited_ratio, raw_shape,
        energy_min, energy_max, normalized_deposited_energy,
        deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
        conditional_on_epl, epl_module, epl_cfg, data_size, batch_size, 
        hidden_dims, lr, max_epochs, max_bad_valid_epochs, cache=None):
    
    classifier = ClassifierHelper(module, dataloader, dataset_name, max_deposited_ratio,
        raw_shape, energy_min, energy_max, normalized_deposited_energy,
        deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies, 
        conditional_on_epl, epl_module, epl_cfg, data_size, batch_size, hidden_dims, lr, max_epochs, max_bad_valid_epochs)

    with torch.enable_grad():
        roc_auc = classifier.train()

    return {"shower_classifier_auc": roc_auc}


def sample_timing(module, dataloader, dataset_name, max_deposited_ratio, raw_shape,
        energy_min, energy_max, normalized_deposited_energy,
        deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
        conditional_on_epl, caloflow_epl, normalize_to_epl, epl_module, epl_cfg, num_samples=100000, 
        batch_sizes=[500, 1000, 5000, 10000, 50000], cache=None):
    
    if not module.use_labels:
        print("Cannot create conditional samples when labels are not provided")
    sample_cfg = {
            "dataset": dataset_name,
            "max_deposited_ratio": max_deposited_ratio,
            "raw_shape": raw_shape,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "normalized_deposited_energy": normalized_deposited_energy,
            "deposited_energy_per_layer": deposited_energy_per_layer,
            "logitspace_voxels": logitspace_voxels,
            "logspace_incident_energies": logspace_incident_energies,
            "conditional_on_epl": conditional_on_epl,
            "caloflow_epl": caloflow_epl,
            "normalize_to_epl": normalize_to_epl,
        }
    times = {}
    
    # Change batch size
    y_dataset = dataloader.dataset
    for bs in batch_sizes:
        for samples in [bs, num_samples]:
            try:
                # Change batch size
                y_loader = get_loader(y_dataset, bs, drop_last=False, pin_memory=True)
                t0 = time.time()
                print("Sampling showers for timing metrics")
                showers, _ = sample_showers(y_loader, module, sample_cfg, samples,
                                            epl_module=epl_module, epl_cfg=epl_cfg)
                t1 = time.time()
                if showers is None:
                    return np.array(0)
                print(f"{samples} conditional samples generated with batch size {bs} in {t1 - t0} seconds")
                ms_per_sample = np.array((1000 * (t1 - t0)) / samples)
                times[f"batch_size:{bs}, num_samples:{samples}"] = ms_per_sample
            except (RuntimeError):
                continue

    return times


def likelihood_ood_acc(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=False, cache=cache
    )


def likelihood_ood_acc_low_dim(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=True, cache=cache
    )
