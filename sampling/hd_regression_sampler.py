from sampling.regression_sampler import RegressionSampler
from models.hd_regression_model import HDRegressionModel
from abc import abstractmethod

class HDRegressionSampler(RegressionSampler):
	def __init__(self, model, batch_size):
		super(HDRegressionSampler, self).__init__(model)

		### Ensure model is high dimensional
		assert isinstance(model, HDRegressionModel)

		### Ensure batch size is correct
		assert batch_size <= model.n
		self.batch_size = batch_size

	@abstractmethod
	def sample(self):
		pass
