import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from helper_functions import mat_norm, subsample
from sampling.regression_sampler import RegressionSampler

class EMOptimiser():
	def __init__(self, sampler):
		### Confirm sampler type
		assert isinstance(sampler, RegressionSampler)
		self.sampler = sampler
		#self.Phi = self.sampler.model.Phi
		#self.B = self.sampler.model.B

	def E_step(self, num_samples):
		return np.array([self.sampler.sample() for i in range(num_samples)])

	def M_step(self, samples):
		M = self.Phi.T @ self.B @ self.Phi
		gamma = 0
		for s in samples[:]:
			gamma += s.T @ M @ s
		gamma /= samples.shape[0]
		return np.diag(np.full(self.sampler.model.d_dash, gamma / np.linalg.norm(self.sampler.mean)**2))

	# Use the EM algorithm to sample from the posterior distribution
	# TODO: Explore the idea of using a convergence threshold
	def sample(self, num_samples, max_iters=50, int_samples=10, alpha_0 = 0.1, threshold = 0.01):
		# initialise A
		A_mean = np.zeros(max_iters)
		A = np.diag(np.full(self.sampler.model.d_dash, alpha_0))
		for i in tqdm(range(max_iters)):
			A_mean[i] = np.linalg.norm(A)
			samples = self.E_step(int_samples)
			A = self.M_step(samples)
		return A_mean, self.E_step(num_samples) + self.sampler.mean