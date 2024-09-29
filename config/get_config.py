import os
import json

from .generalized_autoencoder import GAE_CFG_MAP
from .density_estimator import DE_CFG_MAP
from .shared_config import get_shared_config
from .intrinsic_dimension import get_dim_config


_VALID_DATASETS = ["mnist", "fashion-mnist", "cifar10", "svhn", "celeba", "sphere", "photons1",
                   "pions1", "electrons2", "electrons3", "random-gaussian", "random-uniform"]
_VALID_ESTIMATORS = ["corrint", "danco", "ess", "fishers", "knn", "lpca", "mada", "mind_ml", 
                     "mle", "mom", "tle", "twonn"]

def get_single_config(dataset, model, gae, standalone, epl=False):
    assert dataset in _VALID_DATASETS, \
        f"Unknown dataset {dataset}"

    cfg_map = GAE_CFG_MAP if gae else DE_CFG_MAP
    base_config = cfg_map["base"](dataset, standalone, epl=epl)

    try:
        model_config_function = cfg_map[model]
    except KeyError:
        cfg_map.pop("base")
        raise ValueError(
            f"Invalid model {model} for {'GAE' if gae else 'DE'}. "
            + f"Valid choices are {cfg_map.keys()}."
        )

    return {
        **base_config,

        "dataset": dataset,
        "model": model,

        "gae": gae,

        **model_config_function(dataset, standalone)
    }


def get_configs(dataset, generalized_autoencoder, density_estimator):
    gae_cfg = get_single_config(dataset, generalized_autoencoder, True, False)
    de_cfg = get_single_config(dataset, density_estimator, False, False)

    shared_cfg = get_shared_config(dataset)

    shared_cfg["generalized_autoencoder"] = generalized_autoencoder
    shared_cfg["density_estimator"] = density_estimator

    return gae_cfg, de_cfg, shared_cfg


def get_epl_configs(dataset, epl_model, generalized_autoencoder, density_estimator):
    epl_cfg = get_single_config(dataset, epl_model, False, False, epl=True)
    gae_cfg = get_single_config(dataset, generalized_autoencoder, True, False)
    de_cfg = get_single_config(dataset, density_estimator, False, False)

    shared_cfg = get_shared_config(dataset)

    shared_cfg["epl_model"] = epl_model
    shared_cfg["generalized_autoencoder"] = generalized_autoencoder
    shared_cfg["density_estimator"] = density_estimator

    return epl_cfg, gae_cfg, de_cfg, shared_cfg


def get_dimension_config(dataset, estimator):
    assert dataset in _VALID_DATASETS, \
        f"Unknown dataset {dataset}"
    assert estimator in _VALID_ESTIMATORS, \
        f"Unknown estimator {estimator}"

    base_config = get_dim_config(dataset, estimator)

    return {
        **base_config,

        "dataset": dataset,
        "estimator": estimator,
    }


def load_configs_from_run_dir(run_dir):
    cfgs = []
    
    try:
        with open(os.path.join(run_dir, f"epl_config.json"), "r") as f:
            cfgs.append(json.load(f))
    except:
        print("epl_config.json not found.")

    for cfg_type in ["gae", "de", "shared"]:
        with open(os.path.join(run_dir, f"{cfg_type}_config.json"), "r") as f:
            cfgs.append(json.load(f))

    return cfgs


def load_config_from_run_dir(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    return cfg


def load_epl_configs(args, density_estimator=None):
    if args.load_pretrained_epl:
        with open(os.path.join(args.load_dir, "epl_config.json"), "r") as f:
            epl_cfg = json.load(f)
            
        if args.load_pretrained_gae:
            with open(os.path.join(args.load_dir, "gae_config.json"), "r") as f:
                gae_cfg = json.load(f)

        _, de_cfg, shared_cfg = get_configs(gae_cfg["dataset"], gae_cfg["model"], density_estimator)

        de_cfg["data_dim"] = gae_cfg["latent_dim"]

        cfgs = [epl_cfg, gae_cfg, de_cfg, shared_cfg]

    else:
        cfgs = load_configs_from_run_dir(args.load_dir)

    if args.max_epochs_loaded_epl:
        cfgs[0]["max_epochs"] = args.max_epochs_loaded_epl
    if args.max_epochs_loaded_gae:
        cfgs[1]["max_epochs"] = args.max_epochs_loaded_gae
    if args.max_epochs_loaded_de:
        cfgs[2]["max_epochs"] = args.max_epochs_loaded_de
    if args.max_epochs_loaded:
        cfgs[3]["max_epochs"] = args.max_epochs_loaded

    return cfgs


def load_configs(args, density_estimator=None):
    if args.load_pretrained_gae:
        try:
            with open(os.path.join(args.load_dir, "gae_config.json"), "r") as f:
                gae_cfg = json.load(f)
        except FileNotFoundError:
            with open(os.path.join(args.load_dir, "config.json"), "r") as f:
                gae_cfg = json.load(f)

        _, de_cfg, shared_cfg = get_configs(gae_cfg["dataset"], gae_cfg["model"], density_estimator)

        de_cfg["data_dim"] = gae_cfg["latent_dim"]

        cfgs = [gae_cfg, de_cfg, shared_cfg]

    else:
        cfgs = load_configs_from_run_dir(args.load_dir)

    if args.max_epochs_loaded_gae:
        cfgs[0]["max_epochs"] = args.max_epochs_loaded_gae
    if args.max_epochs_loaded_de:
        cfgs[1]["max_epochs"] = args.max_epochs_loaded_de
    if args.max_epochs_loaded:
        cfgs[2]["max_epochs"] = args.max_epochs_loaded

    return cfgs


def load_config(args):
    cfg = load_config_from_run_dir(args.load_dir)

    if args.max_epochs_loaded:
        cfg["max_epochs"] = args.max_epochs_loaded

    return cfg
