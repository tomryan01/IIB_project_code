import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from helper_functions import mat_norm, subsample
from models.regression_model import RegressionModel

class LeastSquaresSampler():
	def __init__(self, model, sampler, mean, batch_size=None):
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

		# Mock subsamples of Phi and B to test format of sampler
		idx = subsample(self.model.n, self.model.n/batch_size)
		Phi = self.model.generate_Phi(self.model.X, self.model.phi, idx)
		B = self.model.generate_B(self.model.B_i, idx)
		try:
			out = sampler(test_z, test_epsilon, test_theta_0, B, test_A, Phi)
		except:
			raise Exception("Malformed sampling function, should take arguments: z, epsilon, theta_0, B, A, Phi")

		# Ensure sampling function outputs a float
		assert isinstance(out, float), "Expect sampler to be a function outputting a float"

		self.sampler = sampler

		### Ensure mean is of correct dimensions
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean

		### Setup batch size
		if not batch_size:
			self.batch_size = self.model.n
		else:
			self.batch_size = batch_size

	def get_Y(self):
		return self.model.Y

	def get_X(self):
		return self.model.X

	def get_d_dash(self):
		return self.model.d_dash

	def E_step(self, num_samples, Phi, B, add_mean=True):
		samples = np.zeros((num_samples, self.model.d_dash))
		for i in range(num_samples):
			epsilon = np.zeros(self.model.nm)
			for j in range(self.model.n):
				if self.model.m == 1:
					epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
				else:
					epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j])))
			theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))
			samples[i] = minimize(self.sampler, x0 = np.zeros(self.model.d_dash), method='Nelder-Mead', args = (epsilon, theta_0, B, self.model.A, Phi)).x
		if not add_mean:
			return samples
		return samples + self.mean

	def M_step(self, samples, Phi, B):
		M = Phi.T @ B @ Phi
		gamma = 0
		for s in samples[:]:
			gamma += s.T @ M @ s
		gamma /= samples.shape[0]
		return np.diag(np.full(self.model.d_dash, gamma / np.linalg.norm(self.mean)**2))

	# Use the EM algorithm to sample from the posterior distribution
	# TODO: Explore the idea of using a convergence threshold
	def sample(self, num_samples, max_iters=50, int_samples=30, alpha_0 = 0.1, threshold = 0.01):
		# initialise A
		A_mean = np.zeros(max_iters)
		A = np.diag(np.full(self.model.d_dash, alpha_0))
		for i in tqdm(range(max_iters)):
			# compute Phi using a minibatch
			idx = subsample(self.model.n, self.model.n/self.batch_size)
			Phi = self.model.generate_Phi(self.model.X, self.model.phi, idx)
			B = self.model.generate_B(self.model.B_i, idx)
			A_mean[i] = np.linalg.norm(A)
			samples = self.E_step(int_samples, Phi, B)
			A = self.M_step(samples, Phi, B)
		return A_mean, self.E_step(num_samples, Phi, B, add_mean=True)

def minimizer1(z, epsilon, theta_0, B, A, Phi):
	a = 0.5*mat_norm(B, Phi @ z - epsilon)
	b = 0.5*mat_norm(A, z - theta_0)
	return a+b

def minimizer2(z, epsilon, theta_0, B, A, Phi):
	theta_n = theta_0 + np.linalg.inv(A) @ Phi.T @ B @ epsilon
	a = 0.5*mat_norm(B, Phi @ z)
	b = 0.5*mat_norm(A, z - theta_n)
	return a + b