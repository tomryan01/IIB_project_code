import numpy as np
import os
from argparse import ArgumentParser
from neural_net.nn_utils import get_trained_network

from utils.delegates_sghmc import *
from sampling.rate_schedulers import *
from models.model_generators import *

# ranges for each type of tuning
ranges = {
    "batch_size" : [2500, 5000, 7500, 10000],
    "int_samples" : [250, 500, 750, 1000]
}

def main(param, output_path, config_file, problem):
    # read from config
    config = configparser.ConfigParser()
    config.read(config_file)
    batch_size = config.getint('sampling.sghmc', 'batch_size')
    epsilon = config.getfloat('sampling.sghmc', 'epsilon')
    gamma = config.getfloat('sampling.sghmc', 'gamma')
    int_samples = config.getint('sampling.sghmc', 'int_samples')
    num_samples = config.getint('metric.hist', 'num_samples')
    metric_iters = config.getint('metric.hist', 'metric_iters')
    
    # set up params
    params = {
        "batch_size" : batch_size,
        "epsilon" : epsilon,
        "gamma" : gamma,
        "int_samples" : int_samples
    }
    
    if problem == 'nn':
        # set up models and sampler
        print("INFO: Training Network")
        net = get_trained_network(config_file)
        print("INFO: Generating Linear Model")
        model, ld_model = generate_nn_linear_models(config_file, net)
    elif problem == 'ls':
        model, ld_model = generate_synthetic_models(config_file)
        
    # sample for parameter range
    inputs = ranges[param]
    metrics = np.zeros(len(inputs))
    if param == 'batch_size':
        # compute samples and errors
        for i, p in enumerate(inputs):
            print("Computing for batch size = {}".format(p))
            metrics[i] = batch_size_delegate(model, ld_model, p, num_samples, metric_iters, params=params)
    elif param == 'int_samples':
        # compute samples and errors
        for i, p in enumerate(inputs):
            print("Computing for int_samples = {}".format(p))
            metrics[i] = int_samples_delegate(model, ld_model, p, num_samples, metric_iters, params=params)
    else:
        raise Exception("Unknown parameter specified")
    
    # make new folder for outputs
    path = f"{output_path}/{param}"
    if not os.path.exists(path):
        raise Exception(f"{path} directory required, please make it and re-run")
    
    # write results to output file
    output = params.copy()
    output[param] = inputs
    output['metric'] = metrics
    df = pd.DataFrame(output)
    df.to_csv(f'{path}/sghmc_tuning_{param}.csv',index=False)
        
    print("Successfully written output to {}".format(path))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--problem', type=str, default='nn')
    parser.add_argument('--output_path', type=str, default='sghmc_outputs/nn')
    parser.add_argument('--config_file', type=str, default='configs/nn_config.ini')
    args = parser.parse_args()
    main(args.param, args.output_path, args.config_file, args.problem)