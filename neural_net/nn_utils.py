import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import os
import configparser

from neural_net.set_data import set_data
from neural_net.get_data import get_training_data, get_validation_data
from neural_net.mlp import MLP

"""
The following functions are acredited to user: ludwigwinkler, from an answer in the pytorch discussion 
forum here: https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/7
- _del_nested_attr
- extract_weights
- _set_nested_attr
- load_weights
- compute_jacobian
"""


def _del_nested_attr(obj, names):
	"""
	Deletes the attribute specified by the given list of names.
	For example, to delete the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'])
	"""
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		_del_nested_attr(getattr(obj, names[0]), names[1:])
  
def extract_weights(mod):
	"""
	This function removes all the Parameters from the model and
	return them as a tuple as well as their original attribute names.
	The weights must be re-loaded with `load_weights` before the model
	can be used again.
	Note that this function modifies the model in place and after this
	call, mod.parameters() will be empty.
	"""
	orig_params = tuple(mod.parameters())
	# Remove all the parameters in the model
	names = []
	for name, p in list(mod.named_parameters()):
		_del_nested_attr(mod, name.split("."))
		names.append(name)

	'''
		Make params regular Tensors instead of nn.Parameter
	'''
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj, names, value) -> None:
	"""
	Set the attribute specified by the given list of names to value.
	For example, to set the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'], value)
	"""
	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod, names, params):
	"""
	Reload a set of weights so that `mod` can be used again to perform a forward pass.
	Note that the `params` are regular Tensors (that can have history) and so are left
	as Tensors. This means that mod.parameters() will still be empty after this call.
	"""
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, x):
	'''

	@param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
	@param x: input since any gradients requires some input
	@return: either store jac directly in parameters or store them differently

	we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
	'''

	jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
	all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
	load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

	def param_as_input_func(model, x, param):
		load_weights(model, [name], [param]) # name is from the outer scope
		out = model(x)
		return out

	concat_jac = torch.Tensor([])

	for i, (name, param) in enumerate(zip(all_names, all_params)):
		jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param, strict=True if i==0 else False, vectorize=False if i==0 else True)
		concat_jac = torch.concat((concat_jac, jac.squeeze(0).flatten()))

	del jac_model # cleaning up
 
	return concat_jac

def _data_exists(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    paths = config["data.paths"]
    for p in paths:
        if not os.path.exists(p):
            False
    return True

def get_trained_network(config_path, refresh=False):
    # set up config
    config = configparser.ConfigParser()
    config.read(config_path)
    
	# check if data has already been set
    if refresh or not _data_exists(config_path):
        set_data(config_path)
        
    # get training data
    x_train, y_train = get_training_data(config_path)
    x_val, y_val = get_validation_data(config_path)

    # convert to pytorch format
    x_train_t = torch.Tensor(x_train.astype(np.float64)).unsqueeze(0)
    y_train_t = torch.Tensor(y_train.astype(np.float64)).unsqueeze(0)
    x_val_t = torch.Tensor(x_val.astype(np.float64))
    y_val_t = torch.Tensor(y_val.astype(np.float64))
    
    # get training configs
    learning_rate = config.getfloat("network.train", "learning_rate")
    batch_size = config.getint("network.train", "batch_size")
    num_epochs = config.getint("network.train", "num_epochs")
    
    # set up model
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # loss helper function
    def compute_validation_loss():
        model.eval()
        out = model(x_val_t)
        model.train()
        return torch.sum((y_val_t - out) ** 2)

	# train model
    model.train()
    for epoch in range(num_epochs):
		# Shuffle the data for each epoch
        permutation = torch.randperm(x_train_t.shape[0])
        inputs = x_train_t[permutation]
        targets = y_train_t[permutation]

        # Loop over each batch of data
        for i in range(0, inputs.shape[0], batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Validation Loss: {compute_validation_loss():.4f}")
            
    print("Training finished")
 
    return model
    