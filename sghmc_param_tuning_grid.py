import numpy as np
from sampling.sghmc_sampling import SGHMCsampler
from models.model_generators import *
from metrics.covariance_metric_helpers import compute_cov_metric
from metrics.histogram_metric_helpers import compute_full_metric
from neural_net.nn_utils import get_trained_network
import argparse

# metric params
num_samples = 20
metric_iters = 1000

def main(config_file, problem, metric):
    if problem == 'synthetic':
        model, ld_model = generate_synthetic_models(config_file)
    elif problem == 'nn':
        net = get_trained_network(config_file)
        model, ld_model = generate_nn_linear_models(config_file, net)
    else:
        raise Exception("Unknown problem specified")
    
    grid_search_hyperparams(model, ld_model, config_file, metric)
 
def grid_search_hyperparams(model, ld_model, config_file, metric_type):
    # get params from config
    config = configparser.ConfigParser()
    config.read(config_file)
    batch_size = config.getint('sampling.sghmc', 'batch_size')
    int_samples = config.getint('sampling.sghmc', 'int_samples')
    num_samples = config.getint('metric.hist', 'num_samples')
    metric_iters = config.getint('metric.hist', 'metric_iters')
    
    # Define the function to optimize
    def f(gamma, epsilon):
        sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
        # generate samples
        print("Generating samples for gamma = {}, epsilon = {}".format(gamma, epsilon))
        samples = np.zeros((num_samples, model.d_dash))
        for i in range(num_samples):
            if i != 0 and i % 100 == 0:
                print("{} samples completed".format(i))
            samples[i] = sghmc_sampler.sample()

		# compute metric
        print("Computing metric for gamma = {}, epsilon = {}".format(gamma, epsilon))
        if metric_type == 'hist':
            metric, _ = compute_full_metric(samples, ld_model.Hinv, np.zeros(model.d_dash), metric_iters)
        elif metric_type == 'cov':
            metric = compute_cov_metric(samples, np.zeros(model.d_dash), ld_model.Hinv)
        else:
            raise Exception(f"Unknown metric: '{metric_type}'")
        return metric

    # Define the power-of-10 range
    p = 5
    epsilon_range = np.logspace(-2, -4, num=p)
    gamma_range = [32e-3]

    # Perform the grid search
    best_epsilon = None
    best_gamma = None
    best_metric = np.inf
    
    metrics = np.zeros((1,p))

    for i, gamma in enumerate(gamma_range):
        for j, epsilon in enumerate(epsilon_range):
            print("i = {}, j = {}".format(i, j))
            current_metric = f(gamma, epsilon)
            metrics[i][j] = current_metric
            if current_metric < best_metric:
                best_metric = current_metric
                best_epsilon = epsilon
                best_gamma = gamma

    # Print the results
    print("Best epsilon:", best_epsilon)
    print("Best gamma:", best_gamma)
    print("Best metric:", best_metric)
    print("Metrics: ")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/nn_config.ini')
    parser.add_argument('--problem', default='nn')
    parser.add_argument('--metric', default='hist')
    args = parser.parse_args()
    main(args.config_file, args.problem, args.metric)