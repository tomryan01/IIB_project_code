import pandas as pd
import numpy as np
import configparser
from argparse import ArgumentParser

from models.model_generators import *
from neural_net.nn_utils import get_trained_network
from sampling.least_squares_sampler import LeastSquaresSampler
from sampling.rate_schedulers import scheduler_from_string

def main(config_file, output_file, num_samples, problem):
    # read from config
    config = configparser.ConfigParser()
    config.read(config_file)
    batch_size = config.getint('sampling.least-squares', 'batch_size')
    n_iters = config.getint('sampling.least-squares', 'n_iters')
    scheduler = scheduler_from_string(config.get('sampling.least-squares', 'scheduler'))
    rate = config.getfloat('sampling.least-squares', 'rate')
    
    # generate samples
    if problem == 'synthetic':
        model, ld_model = generate_synthetic_models(config_file)
        samples = np.zeros((num_samples, model.d_dash))
        ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(model.d_dash))
        for i in range(num_samples):
            if i != 0 and i % 100 == 0:
                print("{} samples completed".format(i))
            samples[i], _ = ls_sampler.sample(rate=rate, threshold=0, max_iters=n_iters, debug=True, scheduler=scheduler)
    elif problem == 'nn':
        net = get_trained_network(config_file)
        model, ld_model = generate_nn_linear_models(config_file, net)
        ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(model.d_dash))
        samples = np.zeros((num_samples, model.d_dash))
        for i in range(num_samples):
            if i != 0 and i % 100 == 0:
                print("{} samples completed".format(i))
            samples[i], _ = ls_sampler.sample(rate=rate, threshold=0, max_iters=n_iters, debug=True, scheduler=scheduler)
    else:
        raise Exception("Invalid problem setting specified")
        
    # save samples
    data = {f"d{i}" : samples[:,i] for i in range(samples.shape[-1])}
    df = pd.DataFrame(data)
    cov_df = pd.DataFrame(ld_model.Hinv)
    df.to_csv(output_file, index=False)
    cov_df.to_csv(output_file[:-4] + '_cov.csv', index=False)
    print("success!")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/nn_config.ini')
    parser.add_argument('--output_file', type=str, default='test_samples/nn/ls.csv')
    parser.add_argument('--problem', type=str, default='nn')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()
    main(args.config_file, args.output_file, args.num_samples, args.problem)
    
    