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

def marginal_2d(x1, x2, res, size, offset):
    grid = np.ones((res, res))
    dx = size / res
    for i in range(res):
        for j in range(res):
            grid[i][j] += np.sum(np.where(np.logical_and(np.logical_and(offset + i*dx < x1, x1 <= offset + (i+1)*dx), np.logical_and(offset + j*dx < x2, x2 <= offset + (j+1)*dx)), 1, 0))
    print(np.sum(grid))
    return grid / np.sum(grid)

# compute D @ A where D = diag(d), used when diag(d) is too large
def diag_mult(d, A):
    return np.multiply(d, A.T).T