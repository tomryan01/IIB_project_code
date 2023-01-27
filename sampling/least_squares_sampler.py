from models.hd_regression_model import RegressionModel
from sampling.hd_regression_sampler import HDRegressionSampler
from helper_functions import subsample, mat_norm, inv_diag
import numpy as np

class LeastSquaresSampler(HDRegressionSampler):
	def __init__(self, model, batch_size, mean):
		super(LeastSquaresSampler, self).__init__(model, batch_size)

		### Ensure mean is of correct dimensions
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean

	### compute objective function exactly
	def L(self, z, epsilon, theta_0):
		idx = subsample(self.model.n, self.model.n/self.batch_size)
		Phi = self.model.generate_Phi(self.model.X, self.model.phi, idx)
		B = self.model.generate_B(self.model.B_i, idx)
		theta_n = theta_0 + inv_diag(self.model.A) @ Phi.T @ B @ epsilon
		return 0.5*mat_norm(B, Phi @ z) + 0.5*mat_norm(self.model.A, z - theta_n)
	
	# compute grad_L using minibatches
	def stoc_grad_L(self, z, epsilon, theta_0):
		idx = subsample(self.model.n, self.model.n/self.batch_size)
		Phi = self.model.generate_Phi(self.model.X, idx)
		B = self.model.generate_B(self.model.B_i, idx)
		return (self.model.n / self.batch_size) * (z.T @ (Phi.T @ B @ Phi + self.model.A) - theta_0.T @ self.model.A - epsilon.T @ B.T @ Phi)

	def sample(self, threshold=5, rate=0.1, max_iters=100000):
		# compute epsilon
		epsilon = np.zeros(self.model.m * self.batch_size)
		for j in range(self.batch_size):
			if self.model.m == 1:
				epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
			else:
				epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j])))
		
		# compute theta_0
		theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))

		# SGD
		z = theta_0.copy()
		iters = 0
		grad = self.stoc_grad_L(z, epsilon, theta_0)
		while np.linalg.norm(grad) > threshold:
			z -= rate * grad / np.linalg.norm(grad)
			grad = self.stoc_grad_L(z, epsilon, theta_0)
			iters += 1
			if iters > max_iters:
				raise Exception("Maximum number of iterations met for SGD")
		return z

		