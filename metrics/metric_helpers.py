import numpy as np
from scipy.stats import multivariate_normal

def get_bin_index(coord, bottom_left_x, top_right_x, n, m):
	def find_index(x, xmin, xmax, n):
		index = int(((x - xmin) * n) / (xmax - xmin))
		if index == n:
			index -= 1
		return index
	return (find_index(coord[0], bottom_left_x[0], top_right_x[0], n), find_index(coord[1], bottom_left_x[1], top_right_x[1], m))

def make_histogram(X, n, d1, d2, Hinv, mu):
	# require n > 1
	assert n > 1
	
	# compute 2D mean and covariances for the particular dimensions
	mu = np.array([mu[d1], mu[d2]])
	Hinv = np.array([[Hinv[d1,d1], Hinv[d1,d2]],[Hinv[d2,d1], Hinv[d2,d2]]])

	# compute boundary of area to consider
	percentile_point = 2.3263
	bottom_left = [mu[0] - percentile_point * np.sqrt(Hinv[0,0]), mu[1] - percentile_point * np.sqrt(Hinv[1,1])]
	top_right = [mu[0] + percentile_point * np.sqrt(Hinv[0,0]), mu[1] + percentile_point * np.sqrt(Hinv[1,1])]

	# function to check for outliers
	def is_within_rectangle(x):
		return np.all(x >= bottom_left) and np.all(x <= top_right)

	# check if outliers exist
	outliers = []
	for x in X:
		x = np.array([x[d1], x[d2]])
		if not is_within_rectangle(x):
			outliers.append(x)
	if len(outliers) != 0:
		# correct boundary box for outliers
		outliers = np.array(outliers)
		top_right[0] = np.max((top_right[0], np.max(outliers[:,0])))
		top_right[1] = np.max((top_right[1], np.max(outliers[:,1])))
		bottom_left[0] = np.min((bottom_left[0], np.min(outliers[:,0])))
		bottom_left[1] = np.min((bottom_left[1], np.min(outliers[:,1])))

	# compute number of bins along y axis based on scale
	m = np.max((2, int(n*np.abs(bottom_left[1]-top_right[1] / bottom_left[0]-top_right[0])))) # make m at least 2 

	# compute individual bin sizes
	dx = np.abs((bottom_left[0] - top_right[0])) / n
	dy = np.abs((bottom_left[1] - top_right[1])) / m

	# form histogram
	hist = np.zeros((n,m))
	outliers = []
	for x in X:
		x = np.array([x[d1], x[d2]])
		hist[get_bin_index(x, bottom_left, top_right, n, m)] += 1

	# compute coordinates of bin centres
	coords = np.full((n,m,2), bottom_left)
	coords -= np.array([dx/2, dy/2])
	for i in range(n):
		for j in range(m):
			coords[i][j] += np.array([dx*(1+i), dy*(1+j)])

	return hist, coords

def compute_distance_metric(hist, coords, cov, mu):
	"""
	Computes a distance metric between an approximate coordinate distribution defined by hist and coords, and a Gaussian with
	mean mu and covariance Hinv, also considers any outlier samples defined by outliers, which are assumed to lie in regions
	with Gaussian probability 0
	"""
	# ensure histogram and coordinates shape is the same
	assert hist.shape[:2] == coords.shape[:2]

	# compute bin size (assume uniform steps in x and y direction)
	dx = coords[0,0,0] - coords[1,0,0]
	dy = coords[0,0,1] - coords[0,1,1]
	dA = dx*dy

	# normalize histogram
	prob = hist / np.sum(hist)

	# get gaussian random variable
	rv = multivariate_normal(mu, cov)

	# compute metric
	metric = 0
	for i in range(coords.shape[0]):
		for j in range(coords.shape[1]):
			# Euclidean distance
			metric += (prob[i][j]-dA*rv.pdf(coords[i][j]))**2
	
	return np.sqrt(metric)

def calc_expected_distance_metric(samples, mean, cov, N):
	metric = 0
	num_samples = samples.shape[0]
	# scale number of bins according to: https://stats.stackexchange.com/questions/114490/optimal-bin-width-for-two-dimensional-histogram
	# TODO: This is likely also dependent on the scale of the distribution
	num_bins = np.max((2, int(5 * num_samples**(1/4))))
	print("Bin Length of order: {}".format(num_bins))
	d = len(mean)
	for _ in range(N):
		d1 = np.random.randint(0, d)
		d2 = d1
		while d2 == d1:
			d2 = np.random.randint(0, d)

		mu = mean
		Hinv = cov
		hist, coords = make_histogram(samples, num_bins, d1, d2, Hinv, mu)
		metric += compute_distance_metric(hist, coords, np.array([[Hinv[d1,d1], Hinv[d1,d2]],[Hinv[d2,d1], Hinv[d2,d2]]]), np.array([mu[d1], mu[d2]]))
	
	metric /= N
	return metric

def compute_full_metric(samples, cov, mu, metric_iters):
	metric = 0
	d = mu.shape[0]
	num_samples = samples.shape[0]
	# scale number of bins according to: https://stats.stackexchange.com/questions/114490/optimal-bin-width-for-two-dimensional-histogram
	# TODO: This is likely also dependent on the scale of the distribution
	num_bins = np.max((2, int(5 * num_samples**(1/4))))
	hists = []
	for _ in range(metric_iters):
		d1 = np.random.randint(0, d)
		d2 = d1
		while d2 == d1:
			d2 = np.random.randint(0, d)

		hist, coords = make_histogram(samples, num_bins, d1, d2, cov, mu)
		hists.append(hist)
		metric += compute_distance_metric(hist, coords, np.array([[cov[d1,d1], cov[d1,d2]],[cov[d2,d1], cov[d2,d2]]]), np.array([mu[d1], mu[d2]]))
	
	metric /= metric_iters
	return metric, hists