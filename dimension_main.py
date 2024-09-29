#!/usr/bin/env python3

import random
import argparse
import pprint
import torch
import numpy as np

from config import get_dimension_config, parse_config_arg
from two_step_zoo import get_writer
from dimensions import mle_inverse_singlek, estimate_dimension
from dimensions.datasets import load_data


parser = argparse.ArgumentParser(description="Intrinsic Dimension Estimation Module")

parser.add_argument("--estimator", type=str, help="Instrinsic dimension estimator to use.")
parser.add_argument("--dataset", type=str, help="Dataset to estimate dimension of.")
parser.add_argument("--load-dir", type=str, default="", help="Not used for dimension estimation.")

parser.add_argument("--config", default=[], action="append",
    help="Override config entries. Specify as `key=value`.")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = get_dimension_config(dataset=args.dataset, estimator=args.estimator)
cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}
cfg["device"] = device

pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "cfg" + 10*"-")
pp.pprint(cfg)

torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])
random.seed(cfg["seed"])

data = load_data(cfg)
writer = get_writer(args, cfg=cfg)

# Log and save results
save_dict = vars(args)

dim_est = estimate_dimension(data, cfg)
print(f"Estimated dimension using method {args.estimator}: {dim_est}")
save_dict["dim_est"] = float(dim_est)

tag = args.dataset + "_dimensions"
writer.write_json(tag, save_dict)
