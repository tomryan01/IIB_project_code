import numpy as np
from helper_functions import subsample
from models.regression_model import RegressionModel

class HDRegressionModel(RegressionModel):
	def __init__(self, X, Y, B, A, phi):
		super(HDRegressionModel, self).__init__(X, Y, B, A, phi)
		self.Phi = self.generate_Phi(X, range(self.n))
		self.B = self.generate_B(X, range(self.n))

	# generate Phi (nm x d') from X (n x m) and function phi (d -> m x d')
	def generate_Phi(self, X, idx):
		Phi = np.zeros((len(idx)*self.m, self.d_dash))
		# TODO: Look at more efficient approach of subsampling here
		for i, index in enumerate(idx):
			Phi[i*self.m:(i+1)*self.m] = self.phi(X[index])
		return Phi

	# generate B (nm x nm) from B_values (n x m)
	def generate_B(self, B_values, idx):
		B_values.flatten()
		B = np.zeros((len(idx)*self.m, len(idx)*self.m))
		# TODO: Look at more efficient approach of subsampling here
		for i, index in enumerate(idx):
			for j in range(self.m):
				B[i+j][i+j] = B_values[index+j]
		return B

	# generate stochastic gradient estimate
	def grad_U(self, theta, batch_size):
		idx = subsample(self.n, self.n/batch_size)
		B = self.generate_B(self.B_i, idx)
		Phi = self.generate_Phi(self.X, idx)
		return (self.n / batch_size) * (Phi.T @ B @ Phi @ theta - Phi.T @ B @ self.Y[idx]) + self.A @ theta

	@DeprecationWarning
	def empirical_grad_U(self, theta, h):
		out = np.zeros(self.d_dash)
		for i in range(self.d_dash):
			delta = np.zeros(self.d_dash)
			delta[i] = h
			out[i] = (self.U(theta + h/2) - self.U(theta - h/2)) / h
		return out
