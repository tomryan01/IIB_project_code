from sampling.hd_regression_sampler import HDRegressionSampler
from tqdm import tqdm
import numpy as np

class SGHMCsampler(HDRegressionSampler):
	def __init__(self, model, batch_size, step_size, theta_start, int_samples, mean, Minv=None, epsilon=0.1, gamma=0.01):
		super(SGHMCsampler, self).__init__(model, batch_size)

		### Ensure inputs are correct types and dimensions
		assert isinstance(step_size, float), "step_size must be of type (float), got {}".format(type(step_size))

		# set d
		self.d = len(theta_start)
		
		# if no M is specified, make it an identity with the correct dimensions
		if Minv is not None:
			assert len(Minv.shape) == 2, "M must be a two dimensional array (matrix)"
			assert Minv.shape[0] == Minv.shape[1], "M must be a square matrix"
			assert Minv.shape[0] == self.d, "M must have the same dimensions as theta_start, got {}, expected {}".format(Minv.shape[0], self.d)
			assert np.count_nonzero(Minv - np.diag(np.diagonal(Minv))) == 0, "M must be diagonal"
			self.Minv = Minv
		else:
			self.Minv = np.diag(np.full(self.d, 1))

		### set attributes
		self.theta_start = theta_start
		self.step_size = step_size
		self.epsilon = epsilon
		self.int_samples = int_samples
		self.alpha = self.epsilon * self.Minv # This assumes an identity friction matrix C
		self.gamma = gamma
		self.nu = self.gamma / self.model.n
		self.beta_hat = np.diag(np.full(self.d, 0))

		### Ensure mean is of correct dimensions
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean

		# initialise
		self.theta_t = self.theta_start

	def sample(self):
		V = (self.epsilon ** 2) * self.Minv # covariance matrix needs to be converted from M, since we use v and not r
		v_t = np.random.multivariate_normal(np.zeros(self.d), V)
		theta = self.theta_t.copy()
		v = v_t.copy()

		# inner loop to ensure low correlation between samples
		for i in range(self.int_samples):
			theta += v
			v += -1*self.nu * self.model.grad_U(theta, self.batch_size) -1*self.alpha @ v + np.random.multivariate_normal(np.zeros(self.d), 2*(self.alpha - self.beta_hat)*self.nu)
		
		# update theta_t
		self.theta_t = theta

		# no MH-step required
		return theta - self.mean

	def sampleWithMean(self):
		return self.sample() + self.mean