import numpy as np
import utils
import tifffile
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

image = tifffile.imread('data/resize_img011.tif')
z = np.array(image[290:500,370:560,0])
z_original = uniform_filter(z, size=100)

x_original = np.linspace(0, z_original.shape[1], z_original.shape[1])
y_original = np.linspace(0, z_original.shape[0], z_original.shape[0])

x_original, y_original = np.meshgrid(x_original, y_original)

z1 = z_original[0:30,:]
z2 = z_original[60:90,:]
z3 = z_original[120:150,:]
z4 = z_original[180:210,:]



#z1 = z
z = np.concatenate((z1, z2, z3, z4), axis=0)

plt.imshow(z)
plt.show()
# Example data points
x_stripes = []
y_stripes = []

y_stripes.append(np.linspace(0, 30, 30))

y_stripes.append(np.linspace(60, 90, 30))

y_stripes.append(np.linspace(120, 150, 30))

y_stripes.append(np.linspace(180, 210, 30))

#x_stripes.append(np.linspace(8, 10, 20))


y = np.array(y_stripes).flatten()

x = np.array(np.linspace(0, 190, z.shape[1]))

#x = np.array(np.linspace(-10, 10, z.shape[0]))

# Create interpolation function

x_repeated, y_repeated = np.meshgrid(x, y)

print("2")
# mu = np.array([0, 0])
# sigma = np.array([[5, 0], [0, 5]])




#z = utils.gaussian_2d(x_repeated, y_repeated, mu, sigma)
