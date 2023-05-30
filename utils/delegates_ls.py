"""
Delegate functions for hyperparameter tuning for least squares sampler
"""

import numpy as np
from models.hd_regression_model import HDRegressionModel
from sampling.least_squares_sampler import LeastSquaresSampler
from sampling.rate_schedulers import *

def batch_size_delegate(model : HDRegressionModel, batch_size : int, debug_steps : int=1000, **kwargs) -> np.ndarray:
    params = kwargs["params"]
    n_iters = params["n_iters"]
    scheduler = params["scheduler"]
    rate = params["rate"]
    nesterov = params["nesterov"]
    ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(model.d_dash))
    sample, debug = ls_sampler.sample(
        rate=rate, 
        threshold=0, 
        max_iters=n_iters, 
        debug=True, 
        debug_steps=debug_steps, 
        scheduler=scheduler,
        momentum='nesterov', 
        beta=nesterov
    )
    return sample, debug

def learning_rate_delegate(model : HDRegressionModel, rate : float, debug_steps : int=1000, **kwargs) -> np.ndarray:
    params = kwargs["params"]
    n_iters = params["n_iters"]
    scheduler = params["scheduler"]
    batch_size = params["batch_size"]
    nesterov = params["nesterov"]
    ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(model.d_dash))
    sample, debug = ls_sampler.sample(
        rate=rate, 
        threshold=0, 
        max_iters=n_iters, 
        debug=True, 
        debug_steps=debug_steps, 
        scheduler=scheduler,
        momentum='nesterov', 
        beta=nesterov
    )
    return sample, debug

def nesterov_momentum_delegate(model : HDRegressionModel, nesterov_param : float, debug_steps : int=1000, **kwargs) -> np.ndarray:
    params = kwargs["params"]
    n_iters = params["n_iters"]
    scheduler = params["scheduler"]
    batch_size = params["batch_size"]
    rate = params["rate"]
    scheduler_n_epochs = params["scheduler_n_epochs"]
    ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(model.d_dash))
    sample, debug = ls_sampler.sample(
        rate=rate, 
        threshold=0, 
        max_iters=n_iters, 
        debug=True, 
        debug_steps=debug_steps, 
        scheduler=scheduler,
        momentum='nesterov', 
        beta=nesterov_param,
        scheduler_n_epochs = scheduler_n_epochs
    )
    return sample, debug