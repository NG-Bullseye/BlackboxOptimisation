import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def kernel(a, b, l=1.0):
    sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-.5 * (1 / l) * sqdist)

def f(x):
    return (np.sin(x[:, 0]) + np.sin(x[:, 1])) / 5+0.5

def gradient_vector(x):
    if len(x.shape) == 1 or x.shape[1] == 1:  # Handles 1D arrays or 2D arrays with only one column
        x1 = -np.sin(x[0]) / 5
        gradient = np.array([x1])
    else:
        x1 = -np.sin(x[0]) / 5
        x2 = -np.sin(x[1]) / 5
        gradient = np.array([x1, x2])
    return gradient
def perturbation(x, mu, width, a):
    perturbation = a * np.exp(-np.sum(((x - mu) / width) ** 2, axis=-1)) - \
                   a * np.exp(-np.sum(((x + mu) / width) ** 2, axis=-1)) + 1
    return perturbation

def offset_function(x_train, x_test, gradient_vector_func, PERT_WIDTH, PERT_SCALE):
    offsets = np.zeros((len(x_test), len(x_train)))
    for i, x in enumerate(x_test):
        for j, train_point in enumerate(x_train):
            mu = train_point.reshape(-1, -1)  # ensure `mu` is a 2D array
            width = PERT_WIDTH
            a = PERT_SCALE * gradient_vector_func(mu)
            offsets[i][j] = perturbation(x.reshape(-1, 2), mu, width, a)  # Ensure `x` is a 2D array
    return offsets


def offset_kernel(X_train, X_test, gradient_vector_func, PERT_WIDTH, PERT_SCALE):
    offset_matrix = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        offset_matrix[i, :] = offset_function(X_train, [X_test[i]], gradient_vector_func, PERT_WIDTH, PERT_SCALE)
    return offset_matrix

def predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=3.0, PERT_SCALE=1.0, USE_OFFSET=True):
    K = kernel(X_train, X_train)
    K_star = kernel(X_test, X_train)
    print("\nShape of K:", K.shape, "First few elements:", K[:3, :3])
    print("Shape of K_star:", K_star.shape, "First few elements:", K_star[:3, :3])
    if USE_OFFSET:
        offsetkernel = offset_kernel(X_train, X_test, gradient_vector, PERT_WIDTH, PERT_SCALE)
        mu_star = K_star @ np.linalg.inv(K) @ Y_train.flatten() + offsetkernel.flatten()
    else:
        mu_star = K_star @ np.linalg.inv(K) @ Y_train.flatten()
    var_star = np.diag(kernel(X_test, X_test) - K_star @ np.linalg.inv(K) @ K_star.T)
    print("Shape of mu_star:", mu_star.shape, "First few elements:", mu_star[:3])
    print("Shape of var_star:", var_star.shape, "First few elements:", var_star[:3])
    return mu_star, var_star

if __name__ == '__main__':
    PERT_WIDTH=1.5
    PERT_SCALE=0.15
    n_iterations = 0
    np.random.seed(42)
    INTERVAL=10
    X_train = np.array([[5,6]])#np.random.uniform(0, INTERVAL, size=(1, 2))
    Y_train = f(X_train)
    x1_test = np.linspace(0, INTERVAL, 100)
    x2_test = np.linspace(0, INTERVAL, 100)
    X_test = np.dstack(np.meshgrid(x1_test, x2_test)).reshape(-1, 2)
    mu_star, var_star = predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=PERT_WIDTH, PERT_SCALE=PERT_SCALE)
