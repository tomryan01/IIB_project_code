from models.ld_regression_model import LDRegressionModel
from models.regression_model import RegressionModel
from tqdm import tqdm
import numpy as np

class SGHMCsampler():
	def __init__(self, step_size, model, theta_start, Minv=None, batch_size=None, epsilon=0.1, gamma=0.01):
		### Ensure inputs are correct types and dimensions
		assert isinstance(step_size, float), "step_size must be of type (float), got {}".format(type(step_size))

		# set d
		self.d = len(theta_start)

		# check that model is a linear regression model (could expand in future to different models)
		assert isinstance(model, RegressionModel), "model must be a RegressionModel object"
		assert not isinstance(model, LDRegressionModel), "model must not be a LDRegressionModel object"
		
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
		self.model = model
		self.theta_start = theta_start
		self.step_size = step_size
		self.epsilon = epsilon
		self.alpha = self.epsilon * self.Minv # This assumes an identity friction matrix C
		self.gamma = gamma
		self.nu = self.gamma / self.model.n
		self.beta_hat = np.diag(np.full(self.d, 0))
		if batch_size:
			self.batch_size = batch_size
		else:
			self.batch_size = int(round(self.model.n / 10))

	def sample(self, num_samples, int_samples=20):
		# initialise
		theta_t = self.theta_start
		samples = np.zeros((num_samples, self.d))

		# loop over samples
		for t in tqdm(range(num_samples)):
			V = (self.epsilon ** 2) * self.Minv # covariance matrix needs to be converted from M, since we use v and not r
			v_t = np.random.multivariate_normal(np.zeros(self.d), V)
			theta = theta_t.copy()
			v = v_t.copy()

			# inner loop to ensure low correlation between samples
			for i in range(int_samples):
				theta += v
				v += -1*self.nu * self.model.grad_U(theta, self.batch_size) -1*self.alpha @ v + np.random.multivariate_normal(np.zeros(self.d), 2*(self.alpha - self.beta_hat)*self.nu)

			# no MH-step required
			samples[t] = theta
		
		return samples