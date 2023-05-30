import numpy as np
from utils.helper_functions import subsample, diag_mult
from models.regression_model import RegressionModel

class HDRegressionModel(RegressionModel):
	def __init__(self, X, Y, B, A, phi):
		super(HDRegressionModel, self).__init__(X, Y, B, A, phi)
		self.Phi = self.generate_Phi(X)
		self.regularizer = self.Phi.T @ diag_mult(self.B_i, self.Y)
 
	# generate Phi (nm x d') from X (n x m) and function phi (d -> m x d')
	def generate_Phi(self, X):
		Phi = np.zeros((self.n, self.d_dash))
		for i in range(self.n):
			Phi[i] = self.phi(X[i])
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
	
	# generate Y' (n'm) from Y (nm)
	def generate_Y(self, Y, idx):
		out = np.zeros((len(idx)*self.m))
		# TODO: Look at more efficient approach of subsampling here
		for i, index in enumerate(idx):
			out[i*self.m:(i+1)*self.m] = Y[index*self.m:(index+1)*self.m]
		return out

	def U(self, theta):
		return 0.5 * (theta - self.mean).T @ self.H @ (theta - self.mean)

	# generate stochastic gradient estimate
	def grad_U(self, theta, batch_size):
		idx = subsample(self.n, self.n/batch_size)
		B = self.generate_B(self.B_i, idx)
		Phi = self.Phi[idx]
		Y = self.generate_Y(self.Y, idx)
		return (self.n / batch_size) * (Phi.T @ diag_mult(np.diagonal(B), Phi) @ theta) + self.A @ theta - self.regularizer

	@DeprecationWarning
	def empirical_grad_U(self, theta, h):
		out = np.zeros(self.d_dash)
		for i in range(self.d_dash):
			delta = np.zeros(self.d_dash)
			delta[i] = h
			out[i] = (self.U(theta + h/2) - self.U(theta - h/2)) / h
		return out
