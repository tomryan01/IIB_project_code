"""
Delegate functions for hyperparameter tuning for sghmc sampler
"""

import numpy as np
from models.hd_regression_model import HDRegressionModel
from models.ld_regression_model import LDRegressionModel
from metrics.histogram_metric_helpers import compute_full_metric
from metrics.covariance_metric_helpers import compute_cov_metric
from sampling.sghmc_sampling import SGHMCsampler

def batch_size_delegate(model : HDRegressionModel, ld_model : LDRegressionModel, batch_size : int, num_samples : int, metric_iters : int, **kwargs) -> float:
    # get config params
    params = kwargs["params"]
    epsilon = params["epsilon"]
    gamma = params["gamma"]
    int_samples = params["int_samples"]
    
    # sampling
    sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
    samples = np.zeros((num_samples, model.d_dash))
    for i in range(num_samples):
        if i != 0 and i % 100 == 0:
            print("{} samples completed".format(i))
        samples[i] = sghmc_sampler.sample()
        
    # compute metric
    print("Computing metric for batch_size = {}".format(batch_size))
    metric, _ = compute_full_metric(samples, ld_model.Hinv, np.zeros(ld_model.d_dash), metric_iters)
    return metric

def int_samples_delegate(model : HDRegressionModel, ld_model : LDRegressionModel, int_samples : int, num_samples : int, metric_iters : int, **kwargs) -> float:
    # get config params
    params = kwargs["params"]
    epsilon = params["epsilon"]
    gamma = params["gamma"]
    batch_size = params["batch_size"]
    
    # sampling
    sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
    samples = np.zeros((num_samples, model.d_dash))
    for i in range(num_samples):
        if i != 0 and i % 100 == 0:
            print("{} samples completed".format(i))
        samples[i] = sghmc_sampler.sample()
        
    # compute metric
    print("Computing metric for int_samples = {}".format(int_samples))
    metric, _ = compute_full_metric(samples, ld_model.Hinv, np.zeros(ld_model.d_dash), metric_iters)
    return metric