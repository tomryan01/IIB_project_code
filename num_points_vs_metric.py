import pandas as pd
import numpy as np
from models.hd_regression_model import HDRegressionModel
from models.ld_regression_model import LDRegressionModel
from sampling.sghmc_sampling import SGHMCsampler
from sampling.least_squares_sampler import LeastSquaresSampler
from sampling.rate_schedulers import flat
from sampling.embedding_functions import linear
from metrics.metric_helpers import make_histogram, compute_distance_metric, compute_full_metric
from plotting.plotting_helpers import plot_density
import argparse

def main(file_path, noise_var, prior_var, epsilon, gamma, batch_size, int_samples, metric_iters, sampler, max_iters, rate):
	# get data from file
	data = pd.read_csv(file_path).to_numpy()
	d = data.shape[1] - 1
	n = data.shape[0]
	y = data[:n,d]
	X = data[:n,:d]
	
	# set up models
	A = np.linalg.inv(np.diag(np.full(d, prior_var)))
	model = HDRegressionModel(X, y, np.full(n, noise_var), A, linear)
	ld_model = LDRegressionModel(X, y, np.full(len(X), noise_var), A, linear)

	# set up samplers
	sghmc_sampler = SGHMCsampler(model, batch_size, ld_model.mean, int_samples, ld_model.mean, Minv=np.diag(np.full(d, 1)), gamma=gamma, epsilon=epsilon)
	ls_sampler = LeastSquaresSampler(model, batch_size, np.zeros(d))

	# generate samples
	N = [10, 25, 50, 100, 250, 500, 1000, 2000]
	metrics = []
	for n in N:
		# generate samples
		print("Generating samples for n = {}".format(n))
		samples = np.zeros((n, d))
		for i in range(n):
			if i != 0 and i % 100 == 0:
				print("{} samples completed".format(i))
			if sampler == 'sghmc':
				samples[i] = sghmc_sampler.sample()
			elif sampler == 'least-squares':
				samples[i], _ = ls_sampler.sample(rate=rate, threshold=0, max_iters=max_iters, debug=False, debug_steps=2500, scheduler=flat)
			else:
				raise Exception("Invalid Sampler Chosen")

		# compute metric
		print("Computing metric for n = {}".format(n))
		metric, hists = compute_full_metric(samples, ld_model.Hinv, ld_model.mean, metric_iters)
		metrics.append(metric)
		print("Metric for n = {} is: {}".format(n, metric))

		# generate example histogram figure
		plot_density(hists[-1], show=False)

	print("Done")
	print(metrics)

if __name__ == "__main__":
    # argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', default='data/synthetic_data.csv')
	parser.add_argument('--noise', default=1.)
	parser.add_argument('--prior', default=1.)
	parser.add_argument('--epsilon', default=0.0001)
	parser.add_argument('--gamma', default=0.0001)
	parser.add_argument('--batch_size', default=500)
	parser.add_argument('--int_samples', default=300)
	parser.add_argument('--metric_iters', default=1000)
	parser.add_argument('--rate', default=1e-5)
	parser.add_argument('--max_iters', default=5000)
	parser.add_argument('--sampler', default='sghmc')
	args = parser.parse_args()
	file_path = args.data
	noise_var = args.noise
	prior_var = args.prior
	epsilon = args.epsilon
	gamma = args.gamma
	batch_size = args.batch_size
	int_samples = args.int_samples
	metric_iters = args.metric_iters
	sampler = args.sampler
	max_iters = args.max_iters
	rate = args.rate

	# run main
	main(file_path, noise_var, prior_var, epsilon, gamma, batch_size, int_samples, metric_iters, sampler, max_iters, rate)