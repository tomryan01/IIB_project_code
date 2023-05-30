from models.hd_regression_model import RegressionModel
from sampling.hd_regression_sampler import HDRegressionSampler
from scipy.optimize import minimize
from utils.helper_functions import subsample, mat_norm, inv_diag, diag_mult
import numpy as np
from tqdm import tqdm
import time as t

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
		theta_n = theta_0 + inv_diag(self.model.A) @ diag_mult(self.model.B_i, self.model.Phi).T @ epsilon
		return 0.5*mat_norm(np.diag(self.model.B_i), self.model.Phi @ z) + 0.5*mat_norm(self.model.A, z - theta_n)
	
	# compute grad_L using minibatches
	def stoc_grad_L(self, z, regularizer):
		s = t.time()
		idx = subsample(self.model.n, self.model.n/self.batch_size)
		self.subsample_time += t.time() - s
		s = t.time()
		Phi = self.model.Phi[idx]
		self.phi_gen_time += t.time() - s
		s = t.time()
		B = self.model.generate_B(self.model.B_i, idx)
		out = (self.model.n / self.batch_size) * z.T @ (Phi.T @ diag_mult(np.diagonal(B), Phi)) + z.T @ self.model.A - regularizer
		self.compute_time += t.time() - s
		return out

	def trueSample(self, epsilon, theta_0):
		return np.linalg.inv(self.model.Phi.T @ diag_mult(self.model.B_i, self.model.Phi) + self.model.A) @ (self.model.Phi.T @ diag_mult(self.model.B_i, epsilon) + self.model.A @ theta_0)

	def sample(self, threshold=0.1, rate=0.001, max_iters=100000, debug=False, momentum=None, beta=None, scheduler=None, debug_steps=10000, time=False, timer_error_threshold=None, **kwargs):
		
		print(f"Starting LS sampling, momentum={momentum}, beta={beta}, rate={rate}, max_iters={max_iters}, scheduler={scheduler}")
		if scheduler == None:
			scheduler = self.cosineScheduler
   
		if time: assert debug

		if beta == None and momentum is not None:
			raise "Must set beta if momentum != None"

		time_diff = None
   
		# start timer, but don't include trueSample compute
		if time:
			t1 = t.time()
   
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
  
		if time:
			t2 = t.time()

		if debug:
			print("Computing true sample")
			trueSample = self.trueSample(epsilon, theta_0)
		else:
			trueSample = None
   
		if time:
			t3 = t.time()

		if debug:
			print("Computing regulariser...")

		### Compute regularizer
		regularizer = self.model.A @ theta_0.T + diag_mult(self.model.B_i, self.model.Phi).T @ epsilon.T

		# SGD
		z = np.zeros_like(theta_0)
		diff = np.inf
		diffs = []
		errors = []
		z_values = np.zeros((max_iters, len(theta_0)))
		grad_time = 0
		dz = 0
		grad = np.zeros_like(theta_0)
  
		def compute_grad(z):
			nonlocal grad, grad_time
			s = t.time()
			grad = self.stoc_grad_L(z, regularizer)
			e = t.time()
			grad_time += (e-s)

		if momentum is not None: v = np.zeros_like(theta_0)
		for i in tqdm(range(max_iters)):
			if i % debug_steps == 0:
				#print("INFO: iters = {}, error = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z - trueSample), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
				print("INFO: iters = {}, norm_z = {}, diff = {}, rate = {}".format(i, np.linalg.norm(z), diff, scheduler(i, max_iters, rate, **kwargs)))
				print("INFO: average gradient time = {}".format(grad_time / debug_steps))
				print("INFO: average subsample time = {}, average Phi generation time = {}, average compute time = {}".format(self.subsample_time / debug_steps, self.phi_gen_time / debug_steps, self.compute_time / debug_steps))
				if debug:
					print("COMPLETE: iters = {}, error = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z - trueSample), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
					#print("INFO: iters = {}, norm_z = {}, norm_grad = {}, rate = {}".format(i, np.linalg.norm(z), np.linalg.norm(grad), scheduler(i, max_iters, rate, **kwargs)))
				#return z, (diffs, errors, trueSample, time_diff)
			if momentum == 'classic':
				compute_grad(z)
				v = beta*v + (1 - beta) * grad
				z -= scheduler(i, max_iters, rate, **kwargs) * v
			elif momentum == 'nesterov':
				proj = z + beta*dz 
				compute_grad(proj)
				dz = beta*dz - scheduler(i, max_iters, rate, **kwargs) * grad
				z += dz
			else:
				compute_grad(z)
				z -= grad * scheduler(i, max_iters, rate, **kwargs)
			diff = np.linalg.norm(grad)
			diffs.append(diff)
			if debug:
				errors.append(np.linalg.norm(trueSample - z))
				z_values[i] = z
				if time:
					if errors[-1] < timer_error_threshold and time_diff is None:
						time_diff = t.time() - t3 + t2 - t1
		return z, (diffs, errors, trueSample, time_diff, z_values)

	# draw a sample with no debugging, timing, or logging
	def sample_raw(self, scheduler, rate=0.001, max_iters=100000, momentum=None, beta=None, **kwargs):
		if beta == None and momentum is not None:
			raise "Must set beta if momentum != None"

		# compute epsilon
		epsilon = np.zeros(self.model.m * self.model.n)
		for j in range(self.model.n):
			if self.model.m == 1:
				epsilon[j:(j+1)] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.array([np.array([self.model.B_i[j]])])))
			else:
				epsilon[j*self.model.m:(j+1)*self.model.m] = np.random.multivariate_normal(np.zeros(self.model.m), np.linalg.inv(np.diag(self.model.B_i[j*self.model.m:(j+1)*self.model.m])))

		# compute theta_0
		theta_0 = np.random.multivariate_normal(np.zeros(self.model.d_dash), np.linalg.inv(self.model.A))

		### Compute regularizer
		regularizer = self.model.A @ theta_0.T + diag_mult(self.model.B_i, self.model.Phi).T @ epsilon.T

		# SGD
		z = np.zeros_like(theta_0)
		grad = np.zeros_like(theta_0)
  
		def compute_grad(z):
			nonlocal grad
			grad = self.stoc_grad_L(z, regularizer)

		if momentum is not None: v = np.zeros_like(theta_0)
		for i in range(max_iters):
			if momentum == 'classic':
				compute_grad(z)
				v = beta*v + (1 - beta) * grad
				z -= scheduler(i, max_iters, rate, **kwargs) * v
			elif momentum == 'nesterov':
				proj = z + beta*dz 
				compute_grad(proj)
				dz = beta*dz - scheduler(i, max_iters, rate, **kwargs) * grad
				z += dz
			else:
				compute_grad(z)
				z -= grad * scheduler(i, max_iters, rate, **kwargs)
		return z

		

		