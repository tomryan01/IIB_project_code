import numpy as np
import matplotlib.pyplot as plt
import configparser
from argparse import ArgumentParser

from utils.delegates_ls import *
from sampling.rate_schedulers import *
from sampling.sghmc_sampling import SGHMCsampler
from models.model_generators import *
from neural_net.nn_utils import get_trained_network


def main(config_file, predictive_file, posterior_file, num_samples):
    # read from config
    config = configparser.ConfigParser()
    config.read(config_file)
    beta = config.getfloat('lin-laplace', 'beta')
    batch_size = config.getint('sampling.sghmc', 'batch_size')
    epsilon = config.getfloat('sampling.sghmc', 'epsilon')
    gamma = config.getfloat('sampling.sghmc', 'gamma')
    int_samples = config.getint('sampling.sghmc', 'int_samples')
    
    # get test data
    x_test, _ = get_test_data(config_file)
    x_test_t = torch.Tensor(x_test.astype(np.float64))
    
    # set up models and sampler
    print("INFO: Training Network")
    net = get_trained_network(config_file)
    print("INFO: Generating Linear Model")
    model, ld_model = generate_nn_linear_models(config_file, net)
    sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(model.d_dash, 1)), gamma=gamma, epsilon=epsilon)
    
    posterior_samples = np.zeros((num_samples, model.d_dash))
    predictive_samples = np.zeros((x_test.shape[0], num_samples))
    
    # generate posterior samples
    print("INFO: Generating posterior samples")
    for i in range(num_samples):
        posterior_samples[i] = sghmc_sampler.sample()
        
    # save posterior samples
    data = {f"d{i}" : posterior_samples[:,i] for i in range(posterior_samples.shape[-1])}
    df = pd.DataFrame(data)
    cov_df = pd.DataFrame(ld_model.Hinv)
    df.to_csv(posterior_file, index=False)
    cov_df.to_csv(posterior_file[:-4] + '_cov.csv', index=False)
    print("INFO: Saved posterior samples")

    # generate predictive samples
    print("INFO: Generating predictive samples")
    net_predictions = net(x_test_t).detach().squeeze(-1)
    for i in range(x_test.shape[0]):
        for j in range(num_samples):
            predictive_samples[i,j] = net_predictions[i] + torch.dot(compute_jacobian(net, x_test_t[i]), torch.from_numpy(posterior_samples[j].astype(np.float32))) + np.sqrt(beta)**(-1) * np.random.standard_normal()

    # generate predictive covariance matrix
    print("INFO: Generating Predictive Variances")
    pred_vars = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        pred_vars[i] = compute_jacobian(net, x_test_t[i]).numpy().T @ ld_model.Hinv @ compute_jacobian(net, x_test_t[i]).numpy()
        
    # save predictive covariance matrix
    pred_var_df = pd.DataFrame({"vars": [pred_vars[i] for i in range(len(pred_vars))]})
    pred_var_df.to_csv(predictive_file[:-4] + '_vars.csv', index=False)
    print("INFO: Saved predictive covariance")

    # save predictive samples
    data = {f"x_test{i}" : predictive_samples[i,:] for i in range(predictive_samples.shape[0])}
    df = pd.DataFrame(data)
    df.to_csv(predictive_file, index=False)
    print("INFO: Saved predictive samples")
    
    # save network predictions
    data = {f"predictions" : [net_predictions[i].item() for i in range(net_predictions.shape[0])]}
    df = pd.DataFrame(data)
    df.to_csv(predictive_file[:-4] + '_net_predictions.csv', index=False)
    print("INFO: Saved network predictions")
    print("Success!")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/nn_config.ini')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--predictive_file', type=str, default='test_samples/nn/sghmc_pred_old.csv')
    parser.add_argument('--posterior_file', type=str, default='test_samples/nn/sghmc_posterior_old.csv')
    args = parser.parse_args()
    main(args.config_file, args.predictive_file, args.posterior_file, args.num_samples)