import pandas as pd
import numpy as np
import configparser
from argparse import ArgumentParser

from models.model_generators import *
from neural_net.nn_utils import get_trained_network
from sampling.sghmc_sampling import SGHMCsampler

def main(config_file, output_file, num_samples, problem):
    # read from config
    config = configparser.ConfigParser()
    config.read(config_file)
    batch_size = config.getint('sampling.sghmc', 'batch_size')
    epsilon = config.getfloat('sampling.sghmc', 'epsilon')
    gamma = config.getfloat('sampling.sghmc', 'gamma')
    int_samples = config.getint('sampling.sghmc', 'int_samples')
    
    # get correct model
    if problem == 'nn':
        net = get_trained_network(config_file)
        model, ld_model = generate_nn_linear_models(config_file, net)
    elif problem == 'synthetic':
        model, ld_model = generate_synthetic_models(config_file)
    else:
        raise Exception("Unknown problem specified")
    
    # generate samples
    sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
    samples = np.zeros((num_samples, model.d_dash))
    for i in range(num_samples):
        if i != 0 and i % 100 == 0:
            print("{} samples completed".format(i))
        samples[i] = sghmc_sampler.sample()
        
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
    parser.add_argument('--output_file', type=str, default='test_samples/nn/sghmc.csv')
    parser.add_argument('--problem', type=str, default='nn')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()
    main(args.config_file, args.output_file, args.num_samples, args.problem)
    
    