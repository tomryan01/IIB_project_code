import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from helper_functions import mat_norm
from models.ld_regression_model import LDRegressionModel

#-- ### NOTE ###
#-- This code is currently equivalent to using LeastSquaresSampler with
#-- batch_size = n, it should eventually be changed to compute the true samples
#-- using the actual matrix inversions etc.

class LDLeastSquaresSampler():
	def __init__(self, model, sampler, mean):
		### Ensure Model is Correct
		assert isinstance(model, LDRegressionModel)
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
		try:
			out = sampler(test_z, test_epsilon, test_theta_0, self.model.B, test_A, self.model.Phi)
		except:
			raise Exception("Malformed sampling function, should take arguments: z, epsilon, theta_0, B, A, Phi")

		# Ensure sampling function outputs a float
		assert isinstance(out, float), "Expect sampler to be a function outputting a float"

		self.sampler = sampler

		### Ensure mean is of correct dimensions
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean

	def get_phi(self):
		return self.model.Phi
	
	def get_B(self):
		return self.model.B

	def get_M(self):
		return self.model.M

	def get_Y(self):
		return self.model.Y

	def get_X(self):
		return self.model.X

	def get_d_dash(self):
		return self.model.d_dash

	# for reference, we can compute the log-probability for small dimensional problems to check convergence
	def _log_prob(self):
		return 0.5*mat_norm(self.model.A, self.model.mean) + 0.5*np.log(np.linalg.det(np.diag(np.full(self.model.d_dash, 1)) + (np.linalg.inv(self.model.A) @ self.model.M)))

	# for reference we can calculate gamme directly using the trace for small dimensional problems
	def _true_gamma(self):
		return np.trace(np.linalg.inv(self.model.M + self.model.A) @ self.model.M)

	def E_step(self, num_samples, add_mean=False):
		samples = np.zeros((num_samples, self.model.d_dash))
		for i in range(num_samples):
			epsilon = np.zeros(self.model.nm)
			for j in range(self.model.n):
				if self.model.m == 1:
					epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
				else:
					epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j])))
			theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))
			samples[i] = minimize(self.sampler, x0 = np.zeros(self.model.d_dash), method='Nelder-Mead', args = (epsilon, theta_0, self.model.B, self.model.A, self.model.Phi)).x
			if add_mean:
				samples[i] += self.mean
		return samples

	def M_step(self, samples):
		gamma = 0
		for s in samples[:]:
			gamma += s.T @ self.model.M @ s
		gamma /= samples.shape[0]
		return np.diag(np.full(self.model.d_dash, gamma / np.linalg.norm(self.mean)**2))

	# Use the EM algorithm to sample from the posterior distribution
	def sample(self, num_samples, max_iters=50, int_samples=30, alpha_0 = 0.1, threshold = 0.01):
		# initialise A
		A_mean = np.zeros(max_iters)
		A = np.diag(np.full(self.model.d_dash, alpha_0))
		for i in tqdm(range(max_iters)):
			A_mean[i] = np.linalg.norm(A)
			samples = self.E_step(int_samples)
			A = self.M_step(samples)
		return A_mean, self.E_step(num_samples, add_mean=True)

def minimizer1(z, epsilon, theta_0, B, A, Phi):
	a = 0.5*mat_norm(B, Phi @ z - epsilon)
	b = 0.5*mat_norm(A, z - theta_0)
	return a+b

def minimizer2(z, epsilon, theta_0, B, A, Phi):
	theta_n = theta_0 + np.linalg.inv(A) @ Phi.T @ B @ epsilon
	a = 0.5*mat_norm(B, Phi @ z)
	b = 0.5*mat_norm(A, z - theta_n)
	return a + b