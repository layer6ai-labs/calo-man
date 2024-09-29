"""
The functionality in this script is mainly used to load models for creating the OOD
histograms in `notebooks/ood_histogram.ipynb`, but can also be extended to generally
load and inspect trained models as desired.

For example, one can add more outputs to `load_single_module` / `load_twostep_module`
as desired, or more evaluators can be added here, among other things.
"""
import os
import torch

from config import load_configs_from_run_dir, load_config_from_run_dir
from two_step_zoo import (
    get_two_step_module,
    get_single_module,
    get_loaders_from_config,
    get_trainer,
    get_single_trainer,
    Writer
)
from two_step_zoo.evaluators import Evaluator


def load_run(run_dir):
    if os.path.exists(os.path.join(run_dir, "config.json")):
        return load_single_module(run_dir)
    elif os.path.exists(os.path.join(run_dir, "shared_config.json")):
        return load_twostep_module(run_dir)
    else:
        raise FileNotFoundError(f"{run_dir} has neither `config.json` nor `shared_config.json`")


def get_writer(run_dir, cfg):
    return Writer(
        logdir=run_dir,
        make_subdir=False,
        tag_group=cfg["dataset"]
    )


def load_single_module(run_dir):
    cfg = load_config_from_run_dir(run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["data_root"] = "../data/"
    train_loader, valid_loader, test_loader = get_loaders_from_config(cfg)

    module = get_single_module(
        cfg,
        train_dataset_size=cfg["train_dataset_size"],
        data_dim=cfg["data_dim"],
        data_shape=cfg["data_shape"]
    ).to(device)

    writer = get_writer(run_dir, cfg)

    ckpt_prefix = "gae" if cfg["gae"] else "de"

    trainer = get_single_trainer(
        module=module,
        ckpt_prefix=ckpt_prefix,
        writer=writer,
        cfg=cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        evaluator=None
    )

    try:
        trainer.load_checkpoint("best_valid")
    except FileNotFoundError:
        trainer.load_checkpoint("latest")

    return {
        "module": module,
        "trainer": trainer,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }


def load_twostep_module(run_dir):
    gae_cfg, de_cfg, shared_cfg = load_configs_from_run_dir(run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gae_cfg["data_root"] = "../data/"
    de_cfg["data_root"] = "../data/"
    shared_cfg["data_root"] = "../data/"
    train_loader, valid_loader, test_loader = get_loaders_from_config(shared_cfg)

    two_step_module = get_two_step_module(gae_cfg, de_cfg, shared_cfg).to(device)

    writer = get_writer(run_dir, shared_cfg)

    trainer = get_trainer(
        two_step_module=two_step_module,
        writer=writer,
        gae_cfg=gae_cfg,
        de_cfg=de_cfg,
        shared_cfg=shared_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        gae_evaluator=None,
        de_evaluator=None,
        shared_evaluator=Evaluator(two_step_module, valid_loader=None, test_loader=None),
        load_best_valid_first=True,
        pretrained_gae_path="",
        freeze_pretrained_gae=None
    )

    # NOTE: Checkpoint loaded by default for two step module

    return {
        "module": two_step_module,
        "trainer": trainer,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }
