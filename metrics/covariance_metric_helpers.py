import numpy as np

def compute_empirical_cov(samples : np.ndarray):
    """
    Computes covariance matrix from samples from distribution X and mean
    Assumes a square covariance matrix
    """
    #mean = np.mean(samples, axis=0)
    n = samples.shape[0]
    d = samples.shape[1]
    mean = np.zeros(d)
    Qn = np.zeros((d, d))
    for s in samples:
        Qn += np.outer((s - mean),(s - mean))
    return 1/n * Qn

def compute_cov_metric(samples : np.ndarray, mean : np.ndarray, true_cov : np.ndarray):
    """
    Compute the MSE of the difference between the true covariance matrix and 
    the empirical covariance matrix from a set of samples
    """
    return np.mean((true_cov - compute_empirical_cov(samples, mean))**2)