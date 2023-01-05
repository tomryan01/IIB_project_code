import numpy as np
from tqdm import tqdm
from helper_functions import inv_diag
from models.regression_model import RegressionModel

class HMCsampler():
	def __init__(self, step_size, model, M=None):
		### Ensure inputs are correct types and dimensions
		assert isinstance(step_size, float), "step_size must be of type (float), got {}".format(type(step_size))
		
		# set d
		self.d = model.mean.shape[0]

		# check that model is a linear regression model (could expand in future to different models)
		assert isinstance(model, RegressionModel), "model must be a RegressionModel object"
		
		# if no M is specified, make it an identity with the correct dimensions
		if M is not None:
			assert len(M.shape) == 2, "M must be a two dimensional array (matrix)"
			assert M.shape[0] == M.shape[1], "M must be a square matrix"
			assert M.shape[0] == self.d, "M must have the same dimensions as theta_start, got {}, expected {}".format(M.shape[0], self.d)
			assert np.count_nonzero(M - np.diag(np.diagonal(M))) == 0, "M must be diagonal"
			self.M = M
		else:
			self.M = np.diag(np.full(self.d, 100))

		### set attributes
		self.model = model
		self.theta_start = self.model.mean
		self.step_size = step_size
	
	# function to compute H, the total energy of the system, for MH correction
	def _total_energy(self, theta, r):
		return self.model.U(theta) + 0.5 * r.T @ inv_diag(self.M) @ r

	def get_M(self):
		return self.M

	def get_d(self):
		return self.d

	def sample(self, num_samples, int_samples=20):
		# initialise
		theta_t = self.theta_start
		samples = np.zeros((num_samples, self.d))
		accept = 0

		# loop over samples
		for i in tqdm(range(num_samples)):
			r_t = np.random.multivariate_normal(np.zeros(self.d), self.M)
			theta = theta_t.copy()
			r = r_t - (self.step_size/2) * self.model.grad_U(theta)

			# inner loop to ensure low correlation between samples
			for j in range(int_samples):
				theta += self.step_size * inv_diag(self.M) @ r
				r -= self.step_size * self.model.grad_U(theta)
			r -= (self.step_size/2) * self.model.grad_U(theta)
			theta_hat = theta.copy()
			r_hat = r.copy()

			# MH correction
			u = np.random.uniform(0, 1)
			log_accept = self._total_energy(theta_t, r_t) - self._total_energy(theta_hat, r_hat)
			if np.log(u) <= min(0, log_accept):
				theta_t = theta_hat.copy()
				accept += 1

			# store sample
			samples[i] = theta_t
			
		return accept/len(samples), samples