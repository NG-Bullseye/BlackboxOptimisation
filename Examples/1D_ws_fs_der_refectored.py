import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def expected_improvement(X, mu, sigma, xi=0.01):
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def kernel(a, b, l=1.0):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist / l**2)


def f(x):
    return (np.sin(x) + 1) / 2

QUANTIZATION_FACTOR = 2
def x_discrete(x):
    return np.round(x / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR

def f_discrete(x):
    return f(x)

def offset_scalar(x):
    return np.cos(x)/2


def perturbation(x, mu, width, a, y_offset):
    return a * ((x[:, np.newaxis] - mu[np.newaxis, :]) / width) * np.exp(-((x[:, np.newaxis] - mu[np.newaxis, :]) / width) ** 2) + y_offset


def offset_kernel(x_train, x_test, offset_scalar_func, offset_range, offset_scale):
    mu = x_train
    width = offset_range
    a = offset_scale * offset_scalar_func(mu)
    y_offset = 0
    offsets = perturbation(x_test, mu, width, a, y_offset)

    offset_scalar_all = offset_scalar_func(x_train)
    print("Offset scalar for all train points:", offset_scalar_all)

    offset_matrix = offsets.sum(axis=1)
    return offset_matrix




def predict_and_plot(x_train, y_train, x_test, kernel, f, offset_range=3.0, offset_scale=1.0):
    K = kernel(x_train, x_train)
    K_star = kernel(x_test, x_train)
    offsetkernel = offset_kernel(x_train, x_test, offset_scalar, offset_range=offset_range, offset_scale=offset_scale)
    mu_star = K_star @ np.linalg.inv(K) @ y_train.flatten() + offsetkernel.flatten()
    print("mu_star:", mu_star)
    var_star = np.diag(kernel(x_test, x_test) - K_star @ np.linalg.inv(K) @ K_star.T)

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

    return mu_star, var_star

if __name__ == '__main__':
    OFFSET_RANGE=1
    OFFSET_SCALE=1
    n_iterations = 2
    np.random.seed(42)
    INTERVAL=10
    x_train = x_discrete(  np.random.uniform(0, INTERVAL, 2).reshape(-1, 1))

    y_train = f_discrete(x_train)

    x_test = np.linspace(0, INTERVAL, 100).reshape(-1, 1)

    # predict and plot before the loop
    mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f,offset_range=OFFSET_RANGE,offset_scale=OFFSET_SCALE)

    for iteration in range(n_iterations):
        EI = expected_improvement(x_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
        x_next = x_discrete(x_test[np.argmax(EI)])
        y_next = f_discrete(x_next)
        x_train = np.vstack((x_train, x_next))
        y_train = np.vstack((y_train, y_next))
        print(f"Iteration {iteration+1}: x_next = {x_next[0]}, y_next = {y_next[0]}")

        # predict and plot after each iteration
        mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)
