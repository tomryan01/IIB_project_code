# IIB Project Code - Applying Efficient High Dimensional Sampling Techniques to the Linearized Laplace Method
## Overview
This is the code repository for my masters project titled **Applying Efficient High Dimensional Sampling Techniques to the Linearized Laplace Method**. This code has two methods from which to sample from the posterior distribution of a linear model, namely [least squares optimization sampling](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fabs%2F2210.04994&data=05%7C01%7Ctr452%40universityofcambridgecloud.onmicrosoft.com%7Ccff4a85031ae4beb43e008daac6c12aa%7C49a50445bdfa4b79ade3547b4f3986e9%7C0%7C0%7C638011877405786205%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=z8eZff%2BH58cpxnFNmaqIx6RTaFjGfvBeI7ZHp3DDC74%3D&reserved=0) and Stochastic Gradient Hamiltonian Monte Carlo [(SGHMC)](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fabs%2F1402.4102&data=05%7C01%7Ctr452%40universityofcambridgecloud.onmicrosoft.com%7Ca8658a13077f42d106cf08dab836b5c1%7C49a50445bdfa4b79ade3547b4f3986e9%7C0%7C0%7C638024842788212083%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=j2N3tQMIbLh%2F4XxW%2FXJfdKD5dKeXPYBBFw7NH8SnSnM%3D&reserved=0). There are two problem settings from which to draw posterior samples, a simple synthetic regression problem, and a neural network problem. The neural network problem also allows for predictive samples to be drawn using a sampling based approach to the linearized Laplace method [ref](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fabs%2F2210.04994&data=05%7C01%7Ctr452%40universityofcambridgecloud.onmicrosoft.com%7Ccff4a85031ae4beb43e008daac6c12aa%7C49a50445bdfa4b79ade3547b4f3986e9%7C0%7C0%7C638011877405786205%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=z8eZff%2BH58cpxnFNmaqIx6RTaFjGfvBeI7ZHp3DDC74%3D&reserved=0).

## Synthetic Data Problem
First ensure there is a `/data` folder in the root directory, and run
```
python generate_synthetic_data.py
```
To generate the data, ensure the `[data.paths]` config in `configs/synthetic_config.ini` points to the output csv from running this command. 

Parameter tuning can then be performed on this dataset for the SGHMC and Least Squares Optimization Samplers. First ensure the folder `ls_outputs/{parameter}` or `sghmc_outputs/{parameter}` exists and then run
```
python ls_param_tuning.py --param {parameter} --config_file configs/synthetic_config.ini
python sghmc_param_tuning_linear.py --param {parameter} --config_file configs/synthetic_config.ini --problem synthetic
python sghmc_param_tuning_grid.py --config_file configs/synthetic_config.ini --problem synthetic
```
The possible parameters for the least squares sampler are `rate, n_iters, batch_size, nesterov`, and for the sghmc sampler `batch_size, int_samples`. This will populate a csv file in `ls_outputs/{parameter}` and `sghmc_outputs/{parameter}` which can then be used to perform hyperparameter tuning investigations, examples are shown in `notebooks_synthetic/least_squares_hyperprams.ipynb` and `notebooks_synthetic/sghmc_parameter_tuning.ipynb`. **Note that running jupyter notebooks inside a subdirectory requires the path in `synthetic_config.ini` to be replaced with `../data/synthetic_data.csv`**

To simply generate posterior samples with the hyperparameters specified in the config files, ensure the `test_samples/synthetic` and `test_samples/nn` folders exist and run 
```
python ls_generate_test_samples.py --config_file configs/synthetic_config.ini --problem synthetic --output_file test_samples/synthetic/ls.csv --num_samples 100
python sghmc_generate_test_samples.py --config_file configs/synthetic_config.ini --problem synthetic --output_file test_samples/synthetic/ls.csv --num_samples 100
```
These can then be plotted using the examples in `notebooks_synthetic/synthetic_visualizations.ipynb`, again the same warning regarding the relative paths in the config file applies.

## Neural Network Problem
First run the following command
```
python set_nn_data.py
```
To split the data into train, test, and validation, and to store this in the `data/` folder. Ensure the paths in the `configs/nn_config.ini` file point to the csv files generated using this command. Then ensure the `test_samples/nn` folder exists and generate posterior samples and linearized Laplace predictive samples using
```
python ls_linearized_laplace_sampling.py
python sghmc_linearized_laplace_sampling.py
```
This will populate the `test_samples/nn` folder, these samples can then be investigated with examples in the `notebooks_nn` folder. **Note that running jupyter notebooks inside a subdirectory requires a `../` prefix to be added to all paths in `configs/synthetic_config.ini`**

Hyperparameters can also be tuned by ensuring the `ls_outputs/nn` and `sghmc_outputs/nn` folders exist and using
```
python ls_param_tuning.py --param {parameter}
python sghmc_param_tuning_linear.py --param {parameter}
python sghmc_param_tuning_grid.py
```
With the same possible parameters for the synthetic data problem.
