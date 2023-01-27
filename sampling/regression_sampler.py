from sampling.sampler import Sampler
from models.regression_model import RegressionModel
import numpy as np

class RegressionSampler(Sampler):
	mean: np.ndarray
	
	def __init__(self, model):
		super(RegressionSampler, self).__init__(model)

		### ensure model is a regression model
		assert isinstance(model, RegressionModel)