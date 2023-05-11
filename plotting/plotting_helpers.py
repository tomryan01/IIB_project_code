import matplotlib.pyplot as plt

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