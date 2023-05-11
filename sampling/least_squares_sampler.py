from models.hd_regression_model import RegressionModel
from sampling.hd_regression_sampler import HDRegressionSampler
from scipy.optimize import minimize
from helper_functions import subsample, mat_norm, inv_diag, diag_mult
import numpy as np
from tqdm import tqdm
import time

class LeastSquaresSampler(HDRegressionSampler):
	def __init__(self, model, batch_size, mean):
		super(LeastSquaresSampler, self).__init__(model, batch_size)

		### Ensure mean is of correct dimensions
		assert mean.shape[0] == self.model.d_dash, "Length of mean should be equal to the number of parameters {}, got {}".format(self.model.d_dash, mean.shape[0])
		self.mean = mean
		self.subsample_time = 0
		self.phi_gen_time = 0
		self.compute_time = 0

	### compute objective function exactly
	def L(self, z, epsilon, theta_0):
		theta_n = theta_0 + inv_diag(self.model.A) @ self.model.Phi.T @ self.model.B @ epsilon
		return 0.5*mat_norm(self.model.B, self.model.Phi @ z) + 0.5*mat_norm(self.model.A, z - theta_n)
	
	# compute grad_L using minibatches
	def stoc_grad_L(self, z, regularizer):
		s = time.time()
		idx = subsample(self.model.n, self.model.n/self.batch_size)
		self.subsample_time += time.time() - s
		s = time.time()
		Phi = self.model.generate_Phi(self.model.X, idx)
		self.phi_gen_time += time.time() - s
		s = time.time()
		B = self.model.generate_B(self.model.B_i, idx)
		out = (self.model.n / self.batch_size) * (z.T @ (Phi.T @ diag_mult(np.diagonal(B), Phi) + self.model.A)) - regularizer
		self.compute_time += time.time() - s
		return out

	def trueSample(self, epsilon, theta_0):
		return np.linalg.inv(self.model.Phi.T @ diag_mult(self.model.B_i, self.model.Phi) + self.model.A) @ (self.model.Phi.T @ diag_mult(self.model.B_i, epsilon) + self.model.A @ theta_0)

	def cosineScheduler(self, n, max_iters, max_rate, **kwargs):
		return max_rate * np.cos(n*np.pi/(2*max_iters))

	def sample(self, threshold=0.1, rate=0.001, max_iters=100000, debug=False, momentum=False, beta=None, scheduler=None, debug_steps=10000, **kwargs):
		if scheduler == None:
			scheduler = self.cosineScheduler

		if beta == None and momentum:
			raise "Must set beta if momentum == true"
		
		if debug:
			print("Computing Epsilon...")

		# compute epsilon
		epsilon = np.zeros(self.model.m * self.model.n)
		for j in range(self.model.n):
			if self.model.m == 1:
				epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
			else:
				epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j*self.model.m:(j+1)*self.model.m])))

		# compute theta_0
		theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))

		if debug:
			print("Computing regulariser...")

		### Compute regularizer
		regularizer = theta_0.T @ self.model.A + epsilon.T @ diag_mult(self.model.B_i, self.model.Phi)

		if debug:
			print("Computing true sample")
			trueSample = self.trueSample(epsilon, theta_0)
		else:
			trueSample = None

		# SGD
		z = np.zeros_like(theta_0)#theta_0.copy()
		diff = np.inf
		diffs = []
		errors = []
		grad_time = 0

		if momentum: v = np.zeros_like(theta_0)
		grad = self.stoc_grad_L(z, regularizer)
		for i in tqdm(range(max_iters)):
			if i % debug_steps == 0:
				#print("INFO: iters = {}, error = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z - trueSample), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
				print("INFO: iters = {}, norm_z = {}, diff = {}, rate = {}".format(i, np.linalg.norm(z), diff, scheduler(i, max_iters, rate, **kwargs)))
				print("INFO: average gradient time = {}".format(grad_time / debug_steps))
				print("INFO: average subsample time = {}, average Phi generation time = {}, average compute time = {}".format(self.subsample_time / debug_steps, self.phi_gen_time / debug_steps, self.compute_time / debug_steps))
				grad_time = 0
				self.subsample_time = 0
				self.phi_gen_time = 0
				self.compute_time = 0
				if debug:
					print("INFO: error = {}".format(np.linalg.norm(trueSample - z)))
			if diff <= threshold:
				if debug:
					#print("COMPLETE: iters = {}, error = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z - trueSample), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
					print("INFO: iters = {}, norm_z = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
				return z, (diffs, errors, trueSample)
			if momentum:
				v = beta*v + (1 - beta) * grad
				z -= scheduler(i, max_iters, rate, **kwargs) * v
			else:
				z -= grad * scheduler(i, max_iters, rate, **kwargs)
			diff = np.linalg.norm(grad)
			diffs.append(diff)
			if debug:
				errors.append(np.linalg.norm(trueSample - z))
			s = time.time()
			grad = self.stoc_grad_L(z, regularizer)
			e = time.time()
			grad_time += (e-s)
		return z, (diffs, errors, trueSample)

		

		