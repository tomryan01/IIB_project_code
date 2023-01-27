from abc import ABC, abstractmethod
from models.model import Model

class Sampler(ABC):
	def __init__(self, model):
		### assert model is correct
		assert isinstance(model, Model)
		self.model = model

	@abstractmethod
	def sample(self):
		pass
