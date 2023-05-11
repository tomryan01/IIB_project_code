import numpy as np

def flat(epoch, n_epochs, lrate_max, **kwargs):
	return lrate_max

def steps(epoch, n_epochs, lrate_max, **kwargs):
	n_steps = kwargs["n_steps"]
	delay = n_epochs / n_steps
	k = np.floor(epoch / delay)
	return lrate_max * np.exp(-k)

def stepped_lr(epoch, n_epochs, lrate_max, **kwargs):
	lr_decay = kwargs["lr_decay"]
	lr_step = kwargs["lr_step"]
	lr_min = kwargs["lr_min"]
	steps = np.floor_divide(epoch, lr_step)
	lr = lrate_max * np.power(lr_decay, steps)
	lr = np.maximum(lr, lr_min)
	return lr