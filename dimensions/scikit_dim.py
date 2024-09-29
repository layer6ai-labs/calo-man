import time
import skdim
import numpy as np
import pandas as pd


def estimate_dimension(dataset, cfg):
    if cfg['estimator'] == 'corrint':
        estimator = skdim.id.CorrInt()
    elif cfg['estimator'] == 'danco':
        estimator = skdim.id.DANCo()
    elif cfg['estimator'] == 'ess':
        estimator = skdim.id.ESS(ver=cfg['ver'], d=cfg['d'])
    elif cfg['estimator'] == 'fishers':
        estimator = skdim.id.FisherS()
    elif cfg['estimator'] == 'knn':
        estimator = skdim.id.KNN()        
    elif cfg['estimator'] == 'lpca':
        estimator = skdim.id.lPCA()
    elif cfg['estimator'] == 'mada':
        estimator = skdim.id.MADA()
    elif cfg['estimator'] == 'mind_ml':
        estimator = skdim.id.MiND_ML()
    elif cfg['estimator'] == 'mle':
        estimator = skdim.id.MLE()    
    elif cfg['estimator'] == 'mom':
        estimator = skdim.id.MOM()
    elif cfg['estimator'] == 'tle':
        estimator = skdim.id.TLE()
    elif cfg['estimator'] == 'twonn':
        estimator = skdim.id.TwoNN()
    else:
        raise ValueError(f"Unknown estimator {cfg['estimator']} provided.")

    print(f"Estimating dimension using {cfg['estimator']}.")
    start = time.time()
    dim = estimator.fit(dataset, n_jobs=cfg['n_jobs']).dimension_
    end = time.time()
    print(f"Estimation took {end-start} seconds.")
    return dim