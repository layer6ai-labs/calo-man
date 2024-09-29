"""
This file modified from:
* https://github.com/stat-ml/GeoMLE/blob/master/geomle/mle.py
* https://github.com/ppope/dimensions/blob/master/estimators/mle.py
"""

import torch
import numpy as np
from dimensions.utils import KNNComputerNoCheck, update_nn


def intrinsic_dim_sample_wise_double_mle(k=5, dist=None, asymptotically_unbiased=False):
    """
    Returns Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani

    Input parameters:
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k (or more) nearest neighbors of each point (Optional)

    Returns:
    dimensionality estimates without, and with M-G correction
    """
    dist = dist[:, 1:(k + 1)]
    if not np.all(dist > 0):
        np.save("error_dist.npy", dist)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    denominator = (k - 2) if asymptotically_unbiased else (k - 1)
    d = d.sum(axis=1) / denominator
    inv_mle = d.copy()

    d = 1. / d
    mle = d
    return mle, inv_mle

def mle_inverse_singlek(dataset, k1=10, cfg=None):
    """
    Returns the Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani

    Input parameters:
    dataset      - data
    k1           - minimal number of nearest neighbours (Default = 10)
    cfg          - config dict with "train_batch_size", and "n_workers"

    Returns:
    two dimensionality estimates
    """

    anchor_dataset = dataset

    print("Computing the KNNs")
    # compute the KNN with pytorch
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=k1 + 1).to(cfg["device"])

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=cfg["train_batch_size"], shuffle=False,
                                                num_workers=cfg["n_workers"])
    bootstrap_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=cfg["train_batch_size"], shuffle=False,
                                                   num_workers=cfg["n_workers"])

    update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer, cfg["device"])
    dist = nn_computer.min_dists.cpu().numpy()

    mle_res, inv_mle_res = [], []
    start_k = 3 if cfg["asymptotically_unbiased"] else 2
    for k in range(start_k, k1+1):
        mle_results, invmle_results = intrinsic_dim_sample_wise_double_mle(k, dist, asymptotically_unbiased=cfg['asymptotically_unbiased'])
        mle_res.append(mle_results.mean())
        inv_mle_res.append(1. / invmle_results.mean())

    return mle_res, inv_mle_res
