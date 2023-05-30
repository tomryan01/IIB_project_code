import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from models.hd_regression_model import HDRegressionModel
from models.ld_regression_model import LDRegressionModel
from sampling.embedding_functions import linear
from neural_net.get_data import *
from neural_net.nn_utils import compute_jacobian


def generate_nn_linear_models(config_path : str, model : nn.Module):
    # phi for linear model
    def phi(x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
            x.double()
        out = compute_jacobian(model, x).numpy()
        return out
    
    # get config values
    config = configparser.ConfigParser()
    config.read(config_path)
    d = config.getint('network.info', 'd')
    n = config.getint('network.info', 'n')
    
    # get data
    x_train, y_train = get_training_data(config_path)
    x_train_t = torch.Tensor(x_train.astype(np.float64)).unsqueeze(0)
    
    # set up linear model
    d_dash = len(phi(torch.randn(d)))
    b_i = 1
    A = np.eye(d_dash)
    y_pred = model(x_train_t).squeeze(0).squeeze(-1).detach().numpy()
    reg_model = HDRegressionModel(x_train, y_train.squeeze() - y_pred, np.full(n, b_i), A, phi)
    ld_model = LDRegressionModel(x_train, y_train.squeeze() - y_pred, np.full(n, b_i), A, phi)
    
    print("Model Generated!")
    
    return reg_model, ld_model
    
def generate_synthetic_models(config_path):
    # get config values
    config = configparser.ConfigParser()
    config.read(config_path)
    prior_var = config.getfloat('synthetic.train', 'prior_var')
    noise_var = config.getfloat('synthetic.train', 'noise_var')
    data_path = config.get('data.paths', 'synthetic')
    
    # get data from file
    data = pd.read_csv(data_path).to_numpy()
    
    # get data params
    d = data.shape[1] - 1
    n = data.shape[0]
    y = data[:n,d]
    X = data[:n,:d]
    
    # generate models
    A = np.linalg.inv(np.diag(np.full(d, prior_var)))
    model = HDRegressionModel(X, y, np.full(n, noise_var), A, linear)
    ld_model = LDRegressionModel(X, y, np.full(len(X), noise_var), A, linear)
    
    return model, ld_model