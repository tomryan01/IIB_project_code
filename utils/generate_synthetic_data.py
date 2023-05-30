import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import configparser

def generate_synthetic_data(config_path, x_var):
	# read from config
    config = configparser.ConfigParser()
    config.read(config_path)
    feature_dims = config.getint('synthetic.info', 'd')
    num_data = config.getint('synthetic.info', 'n')
    weight_prior_var = config.getint('synthetic.train', 'prior_var')
    observation_noise_var = config.getint('synthetic.train', 'noise_var')
    
    def zero_mean_norm(d, var):
        return np.random.multivariate_normal(np.zeros(d), np.eye(d) * (1/np.sqrt(var)))
    
    print("generating...")
    
    true_weights = zero_mean_norm(feature_dims, weight_prior_var)
    
    data = np.zeros((num_data, feature_dims + 1))
    for i in tqdm(range(num_data)):
        data[i,:feature_dims] = zero_mean_norm(feature_dims, x_var)
        data[i,feature_dims] = np.dot(true_weights, data[i, :feature_dims]) + np.random.normal(0, (1/np.sqrt(observation_noise_var)))
        
    df = pd.DataFrame(data)
    df.to_csv('synthetic_data.csv', header=False, index=False)

if __name__ == '__main__'():
	parser = ArgumentParser()
	parser.add_argument('--config_file', type=str, default='configs/config.ini')
	parser.add_argument('--x_var', type=float, default=10.)
	args = parser.parse_args()
	generate_synthetic_data(args.config_file, args.x_var)