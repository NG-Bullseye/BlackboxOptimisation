import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, NormalizedKernelMixin

# Define the function
def f(x):
    return -np.square(x)

QUANTIZATION_FACTOR = 2

def f_discrete(x):
    x = np.round(x * (QUANTIZATION_FACTOR*10)) / (QUANTIZATION_FACTOR*10)
    return f(x)

# Define the space where the function will be optimized
x_space = np.linspace(-2, 2, 400).reshape(-1, 1)

# Define the custom kernel
class RecommendationScalarKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, rec_scalar=1.0):
        self.length_scale = length_scale
        self.rec_scalar = rec_scalar

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = cdist(X / length_scale, X / length_scale, metric='sqeuclidean')
        else:
            Y = np.atleast_2d(Y)
            dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')

        K = np.exp(-.5 * dists)
        print("rec_scalar: "+str(self.rec_scalar))
        #K = K * np.exp(-5 * np.abs(self.rec_scalar))
        #K *= np.sign(self.rec_scalar)  # add directionality bias based on the sign of rec_scalar
        return K

    def set_rec_scalar(self, rec_scalar):
        self.rec_scalar = rec_scalar

# Define the GP model
gp = GaussianProcessRegressor(kernel=RecommendationScalarKernel(length_scale=1.0), alpha=1e-6)

# Bayesian Optimization
def bayesian_optimization(gp, x, y, bounds):
    # Fit GP model
    gp.fit(x, y)

    # Define the acquisition function
    def acquisition(x):
        mean, std = gp.predict(x.reshape(-1, 1), return_std=True)
        return -(mean + 1.96 * std)  # Negative Lower Confidence Bound

    # Minimize the negative acquisition function
    result = minimize(acquisition, x0=np.mean(bounds), bounds=[bounds])
    return result.x

# Initial samples
np.random.seed(42) # Set the seed
x_samples = np.random.uniform(-2, 2, size=(1, 1))
y_samples = f_discrete(x_samples)


def df(x):
    return -2 * x  # derivative of f

SAMPELS = 3
for i in range(SAMPELS):  # Perform 5 steps of BO
    new_x = bayesian_optimization(gp, x_samples, y_samples, bounds=(-2, 2))
    print("new_x: "+str(new_x))
    new_rec_scalar = df(new_x)
    print("df(new_x): "+str(new_rec_scalar))

    # Update kernel and re-instantiate GaussianProcessRegressor
    kernel = RecommendationScalarKernel(length_scale=1.0, rec_scalar=new_rec_scalar)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    new_y = f_discrete(new_x)
    x_samples = np.vstack([x_samples, new_x])
    y_samples = np.append(y_samples, new_y)

gp.fit(x_samples, y_samples)
# Plot the function, the prediction and the samples
plt.figure(figsize=(20, 10))
plt.plot(x_space, f_discrete(x_space), 'r:', label=r'$f(x) = -x^2$')
plt.plot(x_samples, y_samples, 'r.', markersize=10, label='Observations')

# Predict using the GP model
y_pred, sigma = gp.predict(x_space, return_std=True)

plt.plot(x_space, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x_space, x_space[::-1]]),
         np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper right')

plt.show()
