import matplotlib.pyplot as plt
import numpy as np

def plot_interpolation(x, y, z, x_new, y_new, z_new, z_original, x_original, y_original):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original data pointstile
    x_repeated = np.tile(x, len(y))
    y_repeated = np.repeat(y, len(x))
    ax.scatter(x_repeated, y_repeated, z, c='r', marker='.', label='Original Data')

    ax.plot_surface(x_original, y_original, z_original,
                    rstride=1, cstride=1, alpha=0.5, cmap='plasma',
                    edgecolor='none', label='True surface')

    # Plot interpolated surface
    X_new, Y_new = np.meshgrid(x_new, y_new)
    ax.plot_surface(X_new, Y_new, z_new, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none', label='Interpolated Surface')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('2D Interpolation')
    ax.legend()

    plt.show()

def gaussian_2d(x, y, mu, sigma):
    """
    Generate a 2D Gaussian distribution.

    Parameters:
    - x (numpy.ndarray): X coordinates.
    - y (numpy.ndarray): Y coordinates.
    - mu (numpy.ndarray): Mean vector [mu_x, mu_y].
    - sigma (numpy.ndarray): Covariance matrix [[sigma_x^2, cov_xy], [cov_xy, sigma_y^2]].

    Returns:
    - gauss (numpy.ndarray): 2D Gaussian distribution.
    """
    # Unpack mean and covariance matrix
    mu_x, mu_y = mu
    sigma_x_sq, cov_xy = sigma[0]
    cov_yx, sigma_y_sq = sigma[1]

    # Compute terms for the 2D Gaussian
    term1 = 1 / (2 * np.pi * np.sqrt(sigma_x_sq * sigma_y_sq))
    term2 = -0.5 / (1 - cov_xy * cov_yx) * ((x - mu_x)**2 / sigma_x_sq - 2 * cov_xy * (x - mu_x) * (y - mu_y) / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq)) + (y - mu_y)**2 / sigma_y_sq)

    # Compute the Gaussian
    gauss = term1 * np.exp(term2)

    return gauss