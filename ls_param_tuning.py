import numpy as np
import os
from argparse import ArgumentParser
from neural_net.nn_utils import get_trained_network

from utils.delegates_ls import *
from sampling.rate_schedulers import *
from models.model_generators import *

# ranges for each type of tuning
ranges = {
    "n_iters" : np.linspace(1000, 10000, num=10),
    "scheduler" : [flat, cosine_annealing, stepped_lr],
    "rate" : [8e-7],
    "batch_size" : np.linspace(1000, 8000, num=8),
    "nesterov" : [0.5, 0.75, 0.9, 0.99]
}

def main(param, output_path, config_file, debug_steps):
    # read from config
    config = configparser.ConfigParser()
    config.read(config_file)
    batch_size = config.getint('sampling.least-squares', 'batch_size')
    n_iters = config.getint('sampling.least-squares', 'n_iters')
    scheduler = scheduler_from_string(config.get('sampling.least-squares', 'scheduler'))
    rate = config.getfloat('sampling.least-squares', 'rate')
    momentum_type = config.get('sampling.least-squares', 'momentum')
    beta = config.getfloat('sampling.least-squares', 'beta')
    
    # warning for momentum type
    if momentum_type != 'nesterov':
        print(f"WARN: Nesterov momentum is used for least-squares tuning, overriding '{momentum_type}' specified in config")
    
    # set up params
    params = {
        "n_iters" : n_iters,
        "scheduler" : scheduler,
        "rate" : rate,
        "batch_size" : batch_size,
        "nesterov" : beta,
        "scheduler_n_epochs" : 10000
    }
    
    # set up models and sampler
    print("INFO: Training Network")
    net = get_trained_network(config_file)

    # sample for parameter range
    inputs = ranges[param]
    print(inputs)
    errors = np.zeros((len(inputs), n_iters))
    if param == 'batch_size':
        # compute samples and errors
        for i, p in enumerate(inputs):
            print("Computing for batch size = {}".format(p))
            _, debug = batch_size_delegate(model, p, debug_steps=debug_steps, params=params)
            errors[i] = debug[1]
            # make new folder for outputs
            path = f"{output_path}/{param}"
            if not os.path.exists(path):
                raise Exception(f"{path} directory required, please make it and re-run")
            output = params.copy()
            output[param] = p
            output['errors'] = errors[i]
            output['scheduler'] = config.get('sampling.least-squares', 'scheduler') # get scheduler as string
            df = pd.DataFrame(output)
            df.to_csv(f'{path}/ls_tuning_{param}_{i}.csv',index=False)
    elif param == 'rate':
        # compute samples and errors
        for i, p in enumerate(inputs):
            print("Computing for learning rate = {}".format(p))
            print("INFO: Generating Linear Model")
            model, _ = generate_nn_linear_models(config_file, net)
            _, debug = learning_rate_delegate(model, p, debug_steps=debug_steps, params=params)
            errors[i] = debug[1]
            # make new folder for outputs
            path = f"{output_path}/{param}"
            if not os.path.exists(path):
                raise Exception(f"{path} directory required, please make it and re-run")
            output = params.copy()
            output[param] = p
            output['errors'] = errors[i]
            output['scheduler'] = config.get('sampling.least-squares', 'scheduler') # get scheduler as string
            df = pd.DataFrame(output)
            df.to_csv(f'{path}/ls_tuning_{param}_{batch_size}_{i}.csv',index=False)
    elif param == 'nesterov':
        # compute samples and errors
        for i, p in enumerate(inputs):
            print("Computing for nesterov parameter = {}".format(p))
            print("INFO: Generating Linear Model")
            model, _ = generate_nn_linear_models(config_file, net)
            _, debug = nesterov_momentum_delegate(model, p, debug_steps=debug_steps, params=params)
            errors[i] = debug[1]
            # make new folder for outputs
            path = f"{output_path}/{param}"
            if not os.path.exists(path):
                raise Exception(f"{path} directory required, please make it and re-run")
            output = params.copy()
            output[param] = p
            output['errors'] = errors[i]
            output['scheduler'] = config.get('sampling.least-squares', 'scheduler') # get scheduler as string
            df = pd.DataFrame(output)
            df.to_csv(f'{path}/ls_tuning_{param}_{i}.csv',index=False)
    else:
        raise Exception("Unknown parameter specified")
        
    print("Successfully written output to {}".format(path))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--param', type=str)
    parser.add_argument('--output_path', type=str, default='ls_outputs')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--config_file', type=str, default='configs/nn_config.ini')
    parser.add_argument('--sampler_debug_steps', type=int, default=1000)
    args = parser.parse_args()
    main(args.param, args.output_path, args.config_file, args.sampler_debug_steps)