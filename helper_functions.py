import numpy as np

# function to compute matrix norm of a vector
def mat_norm(mat, vect):
	return vect.T @ mat @ vect

# efficiently compute inverse of diagonal matrix - avoids np.linalg.inv for M matrix in HMC
def inv_diag(X):
	return X - np.diag(np.diagonal(X)) + np.diag(np.divide(1, np.diagonal(X), where=np.diagonal(X)!=0))

# subsample a number of indexes by a given factor
# credit 4M24 coursework
def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx