import pandas as pd
import configparser

def get_training_data(config_path : str):
	config = configparser.ConfigParser()
	config.read(config_path)
	paths = config["data.paths"]
		
	# get data
	x_train = pd.read_csv(paths['x_train'], header=None).to_numpy()
	y_train = pd.read_csv(paths['y_train'], header=None).to_numpy()
   
	return x_train, y_train

def get_test_data(config_path : str):
	config = configparser.ConfigParser()
	config.read(config_path)
	paths = config["data.paths"]
		
	# get data
	x_test = pd.read_csv(paths['x_test'], header=None).to_numpy()
	y_test = pd.read_csv(paths['y_test'], header=None).to_numpy()
   
	return x_test, y_test

def get_validation_data(config_path : str):
	config = configparser.ConfigParser()
	config.read(config_path)
	paths = config["data.paths"]
		
	# get data
	x_val = pd.read_csv(paths['x_val'], header=None).to_numpy()
	y_val = pd.read_csv(paths['y_val'], header=None).to_numpy()
   
	return x_val, y_val

def get_all_data(config_path : str):
	config = configparser.ConfigParser()
	config.read(config_path)
    
	x_train, y_train = get_training_data(config_path)
	x_test, y_test = get_test_data(config_path)
	x_val, y_val = get_validation_data(config_path)
	return x_train, x_val, x_test, y_train, y_val, y_test