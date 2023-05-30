from argparse import ArgumentParser
import time
from tqdm import tqdm

from models.model_generators import *
from metrics.likelihood_ratio_test import compute_likelihood_ratio
from sampling.rate_schedulers import scheduler_from_string
from sampling.sghmc_sampling import SGHMCsampler
from sampling.least_squares_sampler import LeastSquaresSampler
from neural_net.nn_utils import get_trained_network

def main(config_file, problem, sampler, num_samples):
    # check sampler is right form
    assert sampler == 'sghmc' or sampler == 'ls' or sampler == 'both', "Invalid sampler selected"
    
    # get params
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # get models
    if problem == 'synthetic':
        model, ld_model = generate_synthetic_models(config_file)
    elif problem == 'nn':
        net = get_trained_network(config_file)
        model, ld_model = generate_nn_linear_models(config_file, net)
    else:
        raise Exception("Unknown problem specified")
    
    ls_time = 0
    sghmc_time = 0
    
    # get samples
    if sampler == 'sghmc' or sampler == 'both':
        batch_size = config.getint('sampling.sghmc', 'batch_size')
        int_samples = config.getint('sampling.sghmc', 'int_samples')
        gamma = config.getfloat('sampling.sghmc', 'gamma')
        epsilon = config.getfloat('sampling.sghmc', 'epsilon')
        sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(ld_model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
        
        mean_time = 0
        for _ in tqdm(range(num_samples)):
            t1 = time.time()
            sghmc_sampler.sample_raw()
            t2 = time.time()
            mean_time += t2 - t1
        mean_time /= num_samples
        sghmc_time = mean_time
        
    if sampler == 'ls' or sampler == 'both':
        batch_size = config.getint('sampling.least-squares', 'batch_size')
        rate = config.getfloat('sampling.least-squares', 'rate')
        n_iters = config.getint('sampling.least-squares', 'n_iters')
        scheduler = scheduler_from_string(config.get('sampling.least-squares', 'scheduler'))
        ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(ld_model.d_dash))
        
        mean_time = 0
        for _ in tqdm(range(num_samples)):
            t1 = time.time()
            ls_sampler.sample_raw(rate=rate, threshold=0, max_iters=n_iters, debug=False, scheduler=scheduler)
            t2 = time.time()
            mean_time += t2 - t1
        mean_time /= num_samples
        ls_time = mean_time
        
    # compute ratio
    if sampler == 'sghmc' or sampler == 'both':
        print(f"Average time for SGHMC sampling = {sghmc_time}s")
    if sampler == 'ls' or sampler == 'both':
        print(f"Average time for LS Optimization sampling = {ls_time}s")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/synthetic_config.ini')
    parser.add_argument('--problem', type=str, default='synthetic')
    parser.add_argument('--sampler', type=str, default='sghmc')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    main(args.config_file, args.problem, args.sampler, args.num_samples)