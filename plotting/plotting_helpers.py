import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal

from metrics.histogram_metric_helpers import *

def plot_density(bins, show=False):
    """
    Given a 2D array of bins with values equal to a count,
    this function plots a visual representation of the density.
    """
    # Calculate the extent of the plot
    x_min, x_max = 0, bins.shape[0]
    y_min, y_max = 0, bins.shape[1]

    # Plot the density
    fig, ax = plt.subplots()
    img = ax.imshow(bins, cmap='gray', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add colorbar
    cbar = fig.colorbar(img)
    cbar.set_label('Density')
    
    if show:
        plt.show()
        
    return img

def two_variable_bar_plot(labels, var1, var2, error_min_1=None, error_max_1=None, error_min_2=None, error_max_2=None, 
                          bar_width=0.35, var_labels=('Variable 1', 'Variable 2'), error_labels=None, ylabel='Value', xlabel='Variable', title=None):
    
    # set up error bars
    if error_min_1 is not None or error_max_1 is not None:
        assert error_min_1 is not None and error_max_1 is not None, "Both minimum and maximum error is required"
        error1 = [error_min_1, error_max_1]
    else:
        error1 = None
    if error_min_2 is not None or error_max_2 is not None:
        assert error_min_2 is not None and error_max_2 is not None, "Both minimum and maximum error is required"
        error2 = [error_min_2, error_max_2]
    else:
        error2 = None
    
    # make plot
    fig, ax = plt.subplots()
    x_pos = [i for i, _ in enumerate(labels)]
    ax.bar(x_pos, var1, bar_width, label=var_labels[0], yerr=error1)
    ax.bar([i + bar_width for i in x_pos], var2, bar_width, label=var_labels[1], capsize=5)
    
    ax.errorbar(x_pos, var1, yerr=error1, fmt='none', color='black', elinewidth=1, capsize=5, label=error_labels[0])
    ax.errorbar([i + bar_width for i in x_pos], var2, yerr=error2, fmt='none', color='black', elinewidth=1, capsize=5, label=error_labels[1])
    
    # Add some text for labels, title, and axes ticks
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks([i + bar_width / 2 for i in x_pos])
    ax.set_xticklabels(labels)

    # Add a legend
    ax.legend()
    
    return fig, ax

def samples_and_true_samples(samples : np.ndarray, num_plots : int, true_cov : np.ndarray, mean : np.ndarray, type='ls'):
    d_dash = true_cov.shape[0]
    fig, ax = plt.subplots(num_plots,num_plots)
    test_samples = np.random.multivariate_normal(mean, true_cov, size=500)
    if type == 'ls':
        label = "Least Squares Samples"
    else:
        label="SGHMC Samples"
    for i in range(num_plots):
        for j in range(num_plots):
            a = 1
            b = 1
            while a == b:    
                a = np.random.randint(0, d_dash)
                b = np.random.randint(0, d_dash)
                if a != b:
                    cov = np.array([[true_cov[a][a], true_cov[a][b]],[true_cov[b][a], true_cov[b][b]]])
                    mu = np.array([mean[a], mean[b]])
                    rv = multivariate_normal(mu, cov)
                    ax[i][j].scatter(samples[:,a], samples[:,b], color='Black', marker='+', label=label)
                    ax[i][j].scatter(test_samples[:,a], test_samples[:,b], c='Gray', zorder=-1, label="True Samples")
                    ax[i][j].set_xlabel("x{}".format(a))
                    ax[i][j].set_ylabel("x{}".format(b))
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1.1,
                        top=0.85,
                        wspace=0.4,
                        hspace=0.4)
    fig.subplots_adjust(top=0.8)
    return fig, ax
    
def create_bins_plot(samples : np.ndarray, num_plots : int, true_cov : np.ndarray, mean : np.ndarray):
    d_dash = true_cov.shape[0]
    fig, ax = plt.subplots(num_plots,num_plots)
    for i in range(num_plots):
        for j in range(num_plots):
            a = 1
            b = 1
            while a == b:    
                a = np.random.randint(0, d_dash)
                b = np.random.randint(0, d_dash)
                if a != b:
                    bins, coords = make_histogram(samples, compute_num_bins(samples.shape[0]), a, b, true_cov, mean)
                    bins = bins / np.sum(bins)
                    x_min, x_max = 0, np.max(bins.shape)
                    y_min, y_max = 0, np.max(bins.shape)
                    x_coord_min, x_coord_max = np.min(coords[:,0,0]), np.max(coords[:,0,0])
                    y_coord_min, y_coord_max = np.min(coords[0,:,1]), np.max(coords[0,:,1]) 
                    ax[i][j].imshow(bins, cmap='Blues', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
                    ax[i][j].set_xlabel("x{}".format(a))
                    ax[i][j].set_ylabel("x{}".format(b))
                    ax[i][j].set_xticks((np.array(range(5))+0.5)*x_max/5, np.round(x_coord_min + (np.array(range(5))+0.5)*(x_coord_max-x_coord_min)/5, 2))
                    ax[i][j].set_yticks((np.array(range(5))+0.5)*y_max/5, np.round(y_coord_min + (np.array(range(5))+0.5)*(y_coord_max-y_coord_min)/5, 2))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1.1,
                        top=0.85,
                        wspace=0.05,
                        hspace=0.4)
    imgs = ax[0][0].get_images()
    vmin, vmax = imgs[0].get_clim()
    imgs[0].set_clim(vmin, vmax*0.8)
    for i in range(num_plots):
        for j in range(num_plots):
            ax[i][j].get_images()[0].set_clim(vmin, vmax*0.8)
    fig.colorbar(imgs[0], ax=ax.ravel().tolist())
    return fig, ax

def create_bins_plot_w_true(samples : np.ndarray, true_cov : np.ndarray, mean : np.ndarray, d1=None, d2=None, clim=None):
    d_dash = true_cov.shape[0]
    fig, ax = plt.subplots(1,2)
    if d1 is None and d2 is None:
        a = 1
        b = 1
        while a == b:    
            a = np.random.randint(0, d_dash)
            b = np.random.randint(0, d_dash)
    else:
        a, b = d1, d2
    bins, coords = make_histogram(samples, compute_num_bins(samples.shape[0]), a, b, true_cov, mean)
    bins = bins / np.sum(bins)
    x_min, x_max = 0, np.max(bins.shape)
    y_min, y_max = 0, np.max(bins.shape)
    x_coord_min, x_coord_max = np.min(coords[:,0,0]), np.max(coords[:,0,0])
    y_coord_min, y_coord_max = np.min(coords[0,:,1]), np.max(coords[0,:,1])
    img = ax[0].imshow(bins, cmap='Blues', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    ax[0].set_xlabel("x{}".format(a))
    ax[0].set_ylabel("x{}".format(b))
    ax[0].set_xticks((np.array(range(5))+0.5)*x_max/5, np.round(x_coord_min + (np.array(range(5))+0.5)*(x_coord_max-x_coord_min)/5, 2))
    ax[0].set_yticks((np.array(range(5))+0.5)*y_max/5, np.round(y_coord_min + (np.array(range(5))+0.5)*(y_coord_max-y_coord_min)/5, 2))
    
    # compute bin size (assume uniform steps in x and y direction)
    dx = coords[0,0,0] - coords[1,0,0]
    dy = coords[0,0,1] - coords[0,1,1]
    dA = dx*dy
    
    mu = np.array([mean[a], mean[b]])
    cov = np.array([[true_cov[a,a], true_cov[a,b]],[true_cov[b,a], true_cov[b,b]]])
    rv = multivariate_normal(mu, cov)
    z = np.zeros((coords.shape[0], coords.shape[1]))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            z[i][j] = rv.pdf(coords[i][j])*dA
    
    x_coord_min, x_coord_max = np.min(coords[:,0,0]), np.max(coords[:,0,0])
    y_coord_min, y_coord_max = np.min(coords[0,:,1]), np.max(coords[0,:,1])
    ax[1].imshow(z, cmap='Blues', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    ax[1].set_xlabel("x{}".format(a))
    ax[1].set_ylabel("x{}".format(b))
    ax[1].set_xticks((np.array(range(5))+0.5)*x_max/5, np.round(x_coord_min + (np.array(range(5))+0.5)*(x_coord_max-x_coord_min)/5, 2))
    ax[1].set_yticks((np.array(range(5))+0.5)*y_max/5, np.round(y_coord_min + (np.array(range(5))+0.5)*(y_coord_max-y_coord_min)/5, 2))
    fig.colorbar(img, ax=ax.ravel().tolist())
    
    imgs = ax[0].get_images()
    vmin, vmax = imgs[0].get_clim()
    if clim == None:
        imgs[0].set_clim(vmin, vmax*0.8)
        ax[1].get_images()[0].set_clim(vmin, vmax*0.8)
    else: 
        imgs[0].set_clim(clim[0], clim[1])
        ax[1].get_images()[0].set_clim(clim[0], clim[1])
    ax[0].set_title('Sample Histogram')
    ax[1].set_title('True Distribution')
    
    plt.subplots_adjust(left=0.1,
                    right=0.75,
                    wspace = 0.5)
    return fig, ax

def plot_difference_histogram(samples : np.ndarray, true_cov : np.ndarray, mean : np.ndarray, d1=None, d2=None):
    d_dash = true_cov.shape[0]
    fig, ax = plt.subplots()
    if d1 is None and d2 is None:
        a = 1
        b = 1
        while a == b:    
            a = np.random.randint(0, d_dash)
            b = np.random.randint(0, d_dash)
    else:
        a, b = d1, d2
    sample_bins, coords = make_histogram(samples, compute_num_bins(samples.shape[0]), a, b, true_cov, mean)
    
    # compute bin size (assume uniform steps in x and y direction)
    dx = coords[0,0,0] - coords[1,0,0]
    dy = coords[0,0,1] - coords[0,1,1]
    dA = dx*dy

    prob = sample_bins / np.sum(sample_bins)

	# get gaussian random variable
    mu = np.array([mean[a], mean[b]])
    cov = np.array([[true_cov[a,a], true_cov[a,b]],[true_cov[b,a], true_cov[b,b]]])
    rv = multivariate_normal(mu, cov)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
			# Euclidean distance
            prob[i][j] = (prob[i][j]-dA*rv.pdf(coords[i][j]))**2
    x_min, x_max = 0, np.max(sample_bins.shape)
    y_min, y_max = 0, np.max(sample_bins.shape)
    x_coord_min, x_coord_max = np.min(coords[:,0,0]), np.max(coords[:,0,0])
    y_coord_min, y_coord_max = np.min(coords[0,:,1]), np.max(coords[0,:,1])
    img = ax.imshow(prob, cmap='Reds', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel("x{}".format(a))
    ax.set_ylabel("x{}".format(b))
    ax.set_xticks((np.array(range(5))+0.5)*x_max/5, np.round(x_coord_min + (np.array(range(5))+0.5)*(x_coord_max-x_coord_min)/5, 2))
    ax.set_yticks((np.array(range(5))+0.5)*y_max/5, np.round(y_coord_min + (np.array(range(5))+0.5)*(y_coord_max-y_coord_min)/5, 2))
    img.set_clim(0,0.0001)
    fig.tight_layout()
    fig.colorbar(img)
    return fig, ax