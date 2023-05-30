import numpy as np

def flat(epoch, n_epochs, lrate_max, **kwargs):
	return lrate_max

def stepped_lr(epoch, n_epochs, lrate_max, **kwargs):
	lr_decay = kwargs["lr_decay"]
	lr_step = kwargs["lr_step"]
	lr_min = kwargs["lr_min"]
	steps = np.floor_divide(epoch, lr_step)
	lr = lrate_max * np.power(lr_decay, steps)
	lr = np.maximum(lr, lr_min)
	return lr

def cosine_annealing(epoch, n_epochs, lrate_max, **kwargs):
    return lrate_max * 0.5 * (1 + np.cos(np.pi * epoch / n_epochs))

def scheduler_from_string(scheduler : str):
	if scheduler == 'flat':
		return flat
	elif scheduler == 'stepped_lr':
		return cosine_annealing
	elif scheduler == 'cosine_annealing':
		return cosine_annealing