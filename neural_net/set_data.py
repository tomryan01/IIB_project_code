import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import configparser
from argparse import ArgumentParser

def set_data(config_path : str):
    # get data
    data = fetch_california_housing()
    X, y = data.data, data.target
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=1)
    x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(x_train_raw, y_train_raw, test_size=0.25, random_state=1)

    # normalize
    scalerX = StandardScaler()
    scalerX.fit(x_train_raw)
    scalerY = StandardScaler()
    scalerY.fit(y_train_raw.reshape(-1,1))
    x_train = scalerX.transform(x_train_raw)
    x_test = scalerX.transform(x_test_raw)
    x_val = scalerX.transform(x_val_raw)
    y_train = scalerY.transform(y_train_raw.reshape(-1,1))
    y_test = scalerY.transform(y_test_raw.reshape(-1,1))
    y_val = scalerY.transform(y_val_raw.reshape(-1,1))

    # save to csv
    config = configparser.ConfigParser()
    config.read(config_path)
    paths = config["data.paths"]
    np.savetxt(paths.get('x_train'), x_train, delimiter=',')
    np.savetxt(paths.get('x_test'), x_test, delimiter=',')
    np.savetxt(paths.get('x_val'), x_val, delimiter=',')
    np.savetxt(paths.get('y_train'), y_train, delimiter=',')
    np.savetxt(paths.get('y_test'), y_test, delimiter=',')
    np.savetxt(paths.get('y_val'), y_val, delimiter=',')
    
    # write info to config
    d = x_train_raw.shape[-1]
    n = x_train_raw.shape[0]
    config.set('network.info', 'n', str(n))
    config.set('network.info', 'd', str(d))
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config.ini')
    args = parser.parse_args()
    set_data(args.config_file)