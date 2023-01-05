from models.regression_model import RegressionModel
import numpy as np

class LDRegressionModel(RegressionModel):
	def __init__(self, X, Y, B, A, phi):
		super(LDRegressionModel, self).__init__(X, Y, B, A, phi)
		self.B = self.generate_B(B)
		self.Phi = self.generate_Phi(X, phi)
		self.M = self.Phi.T @ self.B @ self.Phi
		self.H = self.M + self.A
		self.Hinv = np.linalg.inv(self.H)
		self.mean = self.Hinv @ self.Phi.T @ self.B @ self.Y # precompute mean

	# generate Phi (nm x d') from X (n x m) and function phi (d -> m x d')
	def generate_Phi(self, X, phi):
		nm = self.nm
		Phi = np.zeros((nm, self.d_dash))
		for i in range(self.n):
			Phi[i*self.m:(i+1)*self.m] = phi(X[i])
		return Phi

	# generate B (nm x nm) from B_values (n x m)
	def generate_B(self, B_values):
		nm = self.nm
		B_values.flatten()
		B = np.zeros((nm, nm))
		for i in range(nm):
			B[i][i] = B_values[i]
		return B

	def get_mean(self):
		return self.mean

	def U(self, theta):
		return 0.5 * (theta - self.mean).T @ self.H @ (theta - self.mean)

	def grad_U(self, theta):
		return self.H @ (theta - self.mean)