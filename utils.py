import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Define the color map and normalization once, to be used throughout the module
_cmap = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
_norm = colors.Normalize(vmin=0, vmax=9)

def plot_single_grid(grid, title='grid'):
    """
    Plots a single grid using the predefined color map and normalization.
    """
    fig, ax = plt.subplots()
    plot_one(ax, grid, title)
    plt.show()

def plot_one(ax, grid, title):
    """
    Helper function to plot a single grid.
    If grid is None, plots an empty grid with a placeholder.
    """
    if grid is None:
        grid = np.zeros((10, 10))  # Create a 10x10 empty grid
    elif isinstance(grid, list) or isinstance(grid, tuple):  # Convert list to numpy array if necessary
        grid = np.array(grid)
    
    if grid.dtype.kind not in 'fi':  # Check if the data is float or integer
        raise ValueError("Grid contains non-numeric data.")
    
    ax.imshow(grid, cmap=_cmap, norm=_norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1 + len(grid))])
    ax.set_xticks([x-0.5 for x in range(1 + len(grid[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_grids(grids_a, grids_b, titles_a, titles_b):
    """
    Plots two lists of grids with custom titles in two rows and returns the figure.
    """
    num_grids_a = len(grids_a)
    num_grids_b = len(grids_b)
    num_grids = max(num_grids_a, num_grids_b)
    
    fig, axs = plt.subplots(2, num_grids, figsize=(3 * num_grids, 6))
    
    for i in range(num_grids):
        if i < num_grids_a:
            plot_one(axs[0, i], grids_a[i], titles_a[i])
        else:
            axs[0, i].axis('off')  # Turn off the axis if there's no grid to plot
        
        if i < num_grids_b:
            plot_one(axs[1, i], grids_b[i], titles_b[i])
        else:
            axs[1, i].axis('off')  # Turn off the axis if there's no grid to plot
    
    plt.tight_layout()
    return fig

def save_image(fig, save_path):
    """
    Saves the figure to the specified path.
    """
    fig.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory

def load_json(file_path):
    """
    Loads a JSON file from the specified path.
    """
    with open(file_path) as f:
        data = json.load(f)
    return data

def plot_to_array(fig):
    """
    Converts a Matplotlib figure to an RGB numpy array.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    array = np.asarray(buf)
    return array[:, :, :3]  # Return only the RGB channels
