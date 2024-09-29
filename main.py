#!/usr/bin/env python3

import argparse
import pprint
import torch
import time

from config import get_epl_configs, load_epl_configs, parse_config_arg
from two_step_zoo import (
    get_single_module, get_two_step_module, get_trainer, get_single_trainer,
    get_loaders_from_config, get_writer, get_evaluator, get_ood_evaluator
)


parser = argparse.ArgumentParser(description="Three Step Calo Challenge Density Estimator")

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if load-dir not specified.")
parser.add_argument("--epl-model", type=str,
    help="Model for density estimation of energies per layer, given incident energy. Required if load-dir not specified.")
parser.add_argument("--gae-model", type=str,
    help="Model for generalized autoencoding. Required if load-dir not specified.")
parser.add_argument("--de-model", type=str,
    help="Model for density estimation. Required if load-dir not specified.")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Attempt to load the best_valid checkpoint first.")
parser.add_argument("--load-pretrained-epl", action="store_true",
    help="Load pretrained epl from resume-dir.")
parser.add_argument("--load-pretrained-gae", action="store_true",
    help="Load pretrained gae from resume-dir.")
parser.add_argument("--freeze-pretrained-gae", action="store_true",
    help="Freeze the parameters of the pretrained GAE, i.e. do not train them.")

parser.add_argument("--max-epochs-loaded", type=int,
    help="New maximum shared epochs for loaded model.")
parser.add_argument("--max-epochs-loaded-epl", type=int,
    help="New maximum epochs for loaded GAE model.")
parser.add_argument("--max-epochs-loaded-gae", type=int,
    help="New maximum epochs for loaded GAE model.")
parser.add_argument("--max-epochs-loaded-de", type=int,
    help="New maximum epochs for loaded DE model.")

parser.add_argument("--epl-config", default=[], action="append",
    help="Override gae config entries. Specify as `key=value`.")
parser.add_argument("--gae-config", default=[], action="append",
    help="Override gae config entries. Specify as `key=value`.")
parser.add_argument("--de-config", default=[], action="append",
    help="Override de config entries. Specify as `key=value`.")
parser.add_argument("--shared-config", default=[], action="append",
    help="Override shared config entries. Specify as `key=value`.")

parser.add_argument("--only-test", action="store_true",
    help="Only perform a test, no training.")

parser.add_argument("--test-ood", action="store_true",
    help="Perform an OOD test.")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


if args.load_dir:
    epl_cfg, gae_cfg, de_cfg, shared_cfg = load_epl_configs(
        args=args,
        density_estimator=args.de_model if args.de_model else None
    )
    
    if args.load_pretrained_epl:
        # NOTE: When loading a pretrained EPL/GAE, we do not expect to load either de_cfg or shared_cfg
        # NOTE: If GAE is to be loaded, the EPL model must also be loaded      
        if not args.load_pretrained_gae:
            gae_cfg = {**gae_cfg, **dict(parse_config_arg(kv) for kv in args.gae_config)}
        de_cfg = {**de_cfg, **dict(parse_config_arg(kv) for kv in args.de_config)}
        shared_cfg = {**shared_cfg, **dict(parse_config_arg(kv) for kv in args.shared_config)}
else:
    epl_cfg, gae_cfg, de_cfg, shared_cfg = get_epl_configs(
        dataset=args.dataset,
        epl_model=args.epl_model,
        generalized_autoencoder=args.gae_model,
        density_estimator=args.de_model
    )
    # Update configs with required fields for conditional EPL training
    epl_cfg = {**epl_cfg, **dict(parse_config_arg(kv) for kv in args.epl_config)}
    epl_cfg["dataset"] = epl_cfg["dataset"] + "-epl"
    epl_cfg["make_valid_loader"] = shared_cfg["make_valid_loader"]
    epl_cfg["train_batch_size"] = shared_cfg["train_batch_size"]
    epl_cfg["valid_batch_size"] = shared_cfg["valid_batch_size"]
    epl_cfg["test_batch_size"] = shared_cfg["test_batch_size"]
    gae_cfg = {**gae_cfg, **dict(parse_config_arg(kv) for kv in args.gae_config)}
    gae_cfg["conditional_on_epl"] = True # y of datasets will be (E_inc, EPL)
    de_cfg = {**de_cfg, **dict(parse_config_arg(kv) for kv in args.de_config)}
    de_cfg["data_dim"] = gae_cfg["latent_dim"]
    de_cfg["conditional_on_epl"] = True
    shared_cfg = {**shared_cfg, **dict(parse_config_arg(kv) for kv in args.shared_config)}
    shared_cfg["conditional_on_epl"] = True
    shared_cfg["metric_kwargs"].update({"conditional_on_epl": True})


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "-epl_cfg--" + 10*"-")
pp.pprint(epl_cfg)
print(10*"-" + "-gae_cfg--" + 10*"-")
pp.pprint(gae_cfg)
print(10*"-" + "--de_cfg--" + 10*"-")
pp.pprint(de_cfg)
print(10*"-" + "shared_cfg" + 10*"-")
pp.pprint(shared_cfg)

writer = get_writer(args, epl_cfg=epl_cfg, gae_cfg=gae_cfg, de_cfg=de_cfg, shared_cfg=shared_cfg)

# EPL training routine
epl_train_loader, epl_valid_loader, epl_test_loader = get_loaders_from_config(epl_cfg)

epl_module = get_single_module(
    epl_cfg,
    data_dim=epl_cfg["data_dim"],
    data_shape=epl_cfg["data_shape"],
    label_dim=epl_cfg["label_dim"],
    train_dataset_size=epl_cfg["train_dataset_size"]
).to(device)

epl_evaluator = get_evaluator(
    epl_module,
    train_loader=epl_train_loader, valid_loader=epl_valid_loader, test_loader=epl_test_loader,
    valid_metrics=epl_cfg["valid_metrics"],
    test_metrics=epl_cfg["test_metrics"],
    **epl_cfg.get("metric_kwargs", {}),
)

epl_trainer = get_single_trainer(
    module=epl_module,
    ckpt_prefix="epl",
    writer=writer,
    cfg=epl_cfg,
    train_loader=epl_train_loader,
    valid_loader=epl_valid_loader,
    test_loader=epl_test_loader,
    evaluator=epl_evaluator,
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
        
t0 = time.time()
epl_trainer.train(epl_step=True)
t1 = time.time()
print(f"Energy-per-layer model training time: {t1 - t0} seconds")

# Add epl_module and epl_cfg to configs so that they can be passed to metrics easily for sampling at test time
shared_cfg["epl_module"] = epl_module
shared_cfg["epl_cfg"] = epl_cfg
shared_cfg["metric_kwargs"].update({"epl_module": epl_module, "epl_cfg": epl_cfg})

# Two-step training routine
train_loader, valid_loader, test_loader = get_loaders_from_config(shared_cfg)
two_step_module = get_two_step_module(gae_cfg, de_cfg, shared_cfg).to(device)

gae_evaluator = get_evaluator(
    two_step_module.generalized_autoencoder,
    valid_loader=valid_loader,
    test_loader=test_loader,
    train_loader=train_loader,
    valid_metrics=gae_cfg["valid_metrics"],
    test_metrics=gae_cfg["test_metrics"],
    **gae_cfg.get("metric_kwargs", {}),
)
de_evaluator = get_evaluator(
    two_step_module.density_estimator,
    valid_loader=None, test_loader=None, # Loaders must be updated later by the trainer
    train_loader=train_loader,
    valid_metrics=de_cfg["valid_metrics"],
    test_metrics=de_cfg["test_metrics"],
    **de_cfg.get("metric_kwargs", {}),
)

if args.test_ood or "likelihood_ood_acc" in shared_cfg["test_metrics"]:
    shared_evaluator = get_ood_evaluator(
        two_step_module,
        cfg=shared_cfg,
        include_low_dim=True,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loader=train_loader,
        savedir=writer.logdir
    )
else:
    shared_evaluator = get_evaluator(
        two_step_module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=shared_cfg["valid_metrics"],
        test_metrics=shared_cfg["test_metrics"],
        **shared_cfg.get("metric_kwargs", {}),
    )
gae_evaluator.metric_kwargs.update({"epl_module": epl_module}) # For sampling with epl_module
de_evaluator.metric_kwargs.update({"epl_module": epl_module}) # For sampling with epl_module
shared_evaluator.metric_kwargs.update({"epl_module": epl_module})
shared_evaluator.metric_kwargs.update({"epl_cfg": epl_cfg})

trainer = get_trainer(
    two_step_module=two_step_module,
    writer=writer,
    gae_cfg=gae_cfg,
    de_cfg=de_cfg,
    shared_cfg=shared_cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    gae_evaluator=gae_evaluator,
    de_evaluator=de_evaluator,
    shared_evaluator=shared_evaluator,
    load_best_valid_first=args.load_best_valid_first,
    pretrained_gae_path=args.load_dir if args.load_pretrained_gae else "",
    freeze_pretrained_gae=args.freeze_pretrained_gae if args.freeze_pretrained_gae else None,
    only_test=args.only_test
)
t0 = time.time()
trainer.train()
t1 = time.time()
print(f"Two-step training time: {t1 - t0} seconds")
