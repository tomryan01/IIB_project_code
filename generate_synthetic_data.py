import numpy as np
import pandas as pd
from tqdm import tqdm

feature_dims = 200
num_data = 10000
weight_prior_var = 1
observation_noise_var = 1
x_var = 10

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