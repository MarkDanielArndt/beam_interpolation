import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import plot_interpolation
import Dataset

interp_func = interpolate.interp2d(Dataset.x_repeated, Dataset.y_repeated, Dataset.z, kind='linear')

# Define new grid for interpolation
x_new = np.linspace(np.min(Dataset.x), np.max(Dataset.x), 10)
y_new = np.linspace(np.min(Dataset.y), np.max(Dataset.y), 10)

# Perform interpolation
z_new = interp_func(x_new, y_new)

# Plot original data and interpolated surface
plot_interpolation(Dataset.x, Dataset.y, Dataset.z, x_new, y_new, z_new, Dataset.z_original,
                   Dataset.x_original, Dataset.y_original)

