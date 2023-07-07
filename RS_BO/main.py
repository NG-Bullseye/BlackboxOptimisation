import numpy as np
from GaussianProcess_1D_rs_ws_fs import GaussianProcess
def f(x):
    return (np.sin(x) + 1) / 2

def kernel(a, b, l=1.0):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist / l**2)

def offset_scalar(x):
    return np.cos(x)/2

QUANTIZATION_FACTOR = 2
def x_discrete(x):
    return np.round(x / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR
import matplotlib.pyplot as plt

def plot_gp(gp, x_test, f_discrete):
    mu_star, var_star = gp.mu_star, gp.var_star
    x_train, y_train = gp.x_train, gp.y_train

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill(np.concatenate([x_test, x_test[::-1]]),
             np.concatenate([mu_star - 1.9600 * var_star,
                             (mu_star + 1.9600 * var_star)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("Offset")
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    INTERVAL=10
    x_train = x_discrete(np.random.uniform(0, INTERVAL, 2).reshape(-1, 1))
    y_train = f(x_train)
    x_test = np.linspace(0, INTERVAL, 100).reshape(-1, 1)

    gp = GaussianProcess(kernel, offset_scalar, offset_range=1, offset_scale=1,random_seed=42)
    gp.train(x_train, y_train)
    gp.optimize(x_test, f, 1)
    mu_star, var_star = gp.predict(x_test)
    plot_gp(gp, x_test, f)


