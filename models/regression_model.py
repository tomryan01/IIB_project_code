import numpy as np
from helper_functions import subsample

class RegressionModel():
	def __init__(self, X, Y, B, A, phi):
		### Ensure Input Data is Correct
		# Allow 1 or 2 dimensional X and Y. 1 dimensional for 1D x and y, and 2 dimensional for > 1D x and y.
		assert (len(X.shape) == 1) or (len(X.shape) == 2), "Expected X to have 1 or 2 dimensions, got {}".format(X.shape)
		assert (len(Y.shape) == 1) or (len(Y.shape) == 2), "Expected Y to have 1 or 2 dimensions, got {}".format(Y.shape)

		# B can have 1 or 2 dimensions, but its dimensionality must be equal to that of Y
		assert (len(B.shape) == 1 or len(B.shape) == 2), "Expected B to have 1 or 2 dimensions, got {}".format(B.shape)
		assert (len(B.shape) == len(Y.shape)), "Expected dimensionality of B and Y to be the same, got {} and {}".format(B.shape, Y.shape)

		# A is the prior covariance, it should be square and have dimension equal to that of Y
		assert len(A.shape) == 2, "Prior covariance A should be a 2D array (a matrix)"
		assert A.shape[0] == A.shape[1], "Prior covariance A should be square"
		
		# Set up n, and if X is 1D make d = 1
		self.n = X.shape[0]
		try:
			self.d = X.shape[1]
		except:
			self.d = 1

		# set d_dash to be the dimensions of A
		self.d_dash = A.shape[0]

		# Both Y, X, and D should have n as their first dimension
		assert Y.shape[0] == self.n, "Expected first dimensions of X and Y to be equal, got {} and {}".format(X.shape[0], Y.shape[0])
		assert B.shape[0] == self.n, "B should be a 2 dimensional array with element (i) as an array of the diagonals for B_i, but first dimension is {} and expected {}".format(B.shape[0], self.n)

		# If Y is 1D make m = 1
		try:
			self.m = Y.shape[1]
		except:
			self.m = 1

		# Set nm for convenience
		self.nm = self.n * self.m

		# If m > 1, B should be 2D, and have second dimension of length m
		if self.m != 1:
			assert len(B.shape) == 2, "If Y is 2D, then B should be too"
			assert B.shape[1] == self.m, "B should be a 2 dimensional array with element (i) as an array of the diagonals for B_i, but second dimension is {} and expected {}".format(B.shape[1], self.m)

		# Ensure phi takes inputs of length d
		test = np.ones(self.d)
		try:
			out = phi(test)
		except:
			raise Exception("Phi should accept np.ndarray of length {}".format(self.d))

		# Ensure phi outputs (m x d')
		assert out.shape[0] == self.d_dash, "First dimension of phi output should have length equal to the number of parameters, mean.shape[0] = {}, got {}".format(self.d_dash, out.shape[0])
		if self.m == 1:
			assert len(out.shape) == 1, "Output of phi is 2D, but outputs y are only single values"
		else:
			try: 
				assert out.shape[1] == self.m, "Second dimension of phi output should have length equal to the length of the outputs, m = {}, got {}".format(self.m, out.shape[1])
			except:
				raise Exception("Output of phi is 1D, but outputs y are vectors")

		### Set up values
		Y.flatten()
		self.Y = Y
		self.B_i = B
		self.A = A
		self.X = X
		self.phi = phi

	# generate Phi (nm x d') from X (n x m) and function phi (d -> m x d')
	def generate_Phi(self, X, phi, idx):
		nm = self.nm
		Phi = np.zeros((len(idx)*self.m, self.d_dash))
		# TODO: Look at more efficient approach of subsampling here
		for i, index in enumerate(idx):
			Phi[i*self.m:(i+1)*self.m] = phi(X[index])
		return Phi

	# generate B (nm x nm) from B_values (n x m)
	def generate_B(self, B_values, idx):
		nm = self.nm
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
		Phi = self.generate_Phi(self.X, self.phi, idx)
		return (self.n / batch_size) * (Phi.T @ B @ Phi @ theta - Phi.T @ B @ self.Y) + self.A @ theta

	@DeprecationWarning
	def empirical_grad_U(self, theta, h):
		out = np.zeros(self.d_dash)
		for i in range(self.d_dash):
			delta = np.zeros(self.d_dash)
			delta[i] = h
			out[i] = (self.U(theta + h/2) - self.U(theta - h/2)) / h
		return out
