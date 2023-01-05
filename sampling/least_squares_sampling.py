import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from helper_functions import mat_norm
from models.regression_model import RegressionModel

#-- ### NOTE ###
#-- As of now this class will not work, it is not fully implemented

class LeastSquaresSampler():
	def __init__(self, model, sampler, mean):
		### Ensure Model is Correct
		assert isinstance(model, RegressionModel)
		self.model = model

		### Ensure sampling function takes correct inputs 
		# z (d')
		# epsilon (nm)
		# theta_0 (d')
		# B (nm x nm)
		# A (d' x d')
		# Phi (nm x d')
		test_z = np.zeros(self.model.d_dash)
		test_epsilon = np.ones(self.model.nm)
		test_theta_0 = np.ones(self.model.d_dash)
		test_A = np.diag(np.ones(self.model.d_dash))
		# TODO: Mock subsamples of B and Phi to test sampling function
		# 	Presumably the batch size for SGD must be known and universal so that it can be checked that the sampling
		# 	function accepts this n' and not the larger n
		try:
			out = sampler(test_z, test_epsilon, test_theta_0, self.model.B, test_A, self.model.Phi)
		except:
			raise Exception("Malformed sampling function, should take arguments: z, epsilon, theta_0, B, A, Phi")

		# Ensure sampling function outputs a float
		assert isinstance(out, float), "Expect sampler to be a function outputting a float"

		self.sampler = sampler

		### Ensure mean is of correct dimensions
		# NOTE: Here the mean is likely to be computed using MC for a high-dimensional problem
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean

	def get_Y(self):
		return self.model.Y

	def get_X(self):
		return self.model.X

	def get_d_dash(self):
		return self.model.d_dash

	def E_step(self, num_samples):
		samples = np.zeros((num_samples, self.model.d_dash))
		for i in range(num_samples):
			epsilon = np.zeros(self.model.nm)
			for j in range(self.model.n):
				if self.model.m == 1:
					epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
				else:
					epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j])))
			theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))
			# TODO: Explore the use of different minimisation methods, i.e. SGD
			samples[i] = minimize(self.sampler, x0 = np.zeros(self.model.d_dash), method='Nelder-Mead', args = (epsilon, theta_0, self.model.B, self.model.A, self.model.Phi)).x
		return samples

	def M_step(self, samples):
		gamma = 0
		for s in samples[:]:
			gamma += s.T @ self.model.M @ s
		gamma /= samples.shape[0]
		return np.diag(np.full(self.model.d_dash, gamma / np.linalg.norm(self.mean)**2))

	# Use the EM algorithm to sample from the posterior distribution
	# TODO: Explore the idea of using a convergence threshold
	def sample(self, num_samples, max_iters=50, int_samples=30, alpha_0 = 0.1, threshold = 0.01):
		# initialise A
		A_mean = np.zeros(max_iters)
		A = np.diag(np.full(self.model.d_dash, alpha_0))
		for i in tqdm(range(max_iters)):
			A_mean[i] = np.linalg.norm(A)
			samples = self.E_step(int_samples)
			A = self.M_step(samples)
		# TODO: this will return a zero-mean result, need to determine how to include mean for SGD
		return A_mean, self.E_step(num_samples)

def minimizer1(z, epsilon, theta_0, B, A, Phi):
	a = 0.5*mat_norm(B, Phi @ z - epsilon)
	b = 0.5*mat_norm(A, z - theta_0)
	return a+b

def minimizer2(z, epsilon, theta_0, B, A, Phi):
	theta_n = theta_0 + np.linalg.inv(A) @ Phi.T @ B @ epsilon
	a = 0.5*mat_norm(B, Phi @ z)
	b = 0.5*mat_norm(A, z - theta_n)
	return a + b