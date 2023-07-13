import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class OptimizationLibrary:
    def __init__(self, offset_range=3.0, offset_scale=1.0, protection_width=1.0):
        self.offset_range = offset_range
        self.offset_scale = offset_scale
        self.protection_width = protection_width

    def expected_improvement(self, X, mu, sigma, xi=0.01):
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def kernel(self, a, b, l=1.0):
        sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist / l**2)

    def f(self, x):
        return (np.sin(x) + 1) / 2

    def offset_scalar(self, x):
        return np.cos(x) / 2

    def perturbation(self, x, mu, width, a, y_offset):
        perturbation = a * ((x - mu) / width) * np.exp(-((x - mu) / width) ** 2) + y_offset
        return perturbation

    def parabola_matrix(self, x, mu):
        return ((x - mu.T) ** 2 / self.protection_width)

    def offset_function(self, x_train, x_test, offset_scalar_func):
        offset_scalar_all = offset_scalar_func(x_train)
        print("Offset scalar for all train points:", offset_scalar_all)
        offsets = np.zeros_like(x_test)
        x_train = x_train.reshape(-1)

        for i, mu in enumerate(x_train):
            width = self.offset_range
            a = self.offset_scale * offset_scalar_all[i]
            y_offset = 0
            protection_term = self.parabola_matrix(x_test, mu)
            offsets += self.perturbation(x_test, mu, width, a, y_offset) * protection_term

        return offsets

    def offset_kernel(self, x_train, x_test, offset_scalar_func):
        offset_scalar_all = offset_scalar_func(x_train)
        print("Offset scalar for all train points:", offset_scalar_all)
        offset_matrix = self.offset_function(x_train, x_test, offset_scalar_func)
        return offset_matrix

    def predict_and_plot(self, x_train, y_train, x_test, kernel, f):
        K = kernel(x_train, x_train)
        K_star = kernel(x_test, x_train)
        offsetkernel = self.offset_kernel(x_train, x_test, self.offset_scalar)
        jitter = 1e-6
        K += np.eye(K.shape[0]) * jitter
        mu_star = K_star @ np.linalg.inv(K) @ y_train.flatten() + offsetkernel
        print("mu_star:", mu_star)
        var_star = np.diag(kernel(x_test, x_test)) - np.einsum("ij,ij->i", K_star @ np.linalg.inv(K), K_star)
        plt.figure(figsize=(12, 8))
        plt.plot(x_test, f(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
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
    OFFSET_RANGE = 0.2
    OFFSET_SCALE = 0.5
    n_iterations = 15
    np.random.seed(42)
    INTERVAL = 10
    x_train = np.random.uniform(0, INTERVAL, 2).reshape(-1, 1)
    y_train = OptimizationLibrary().f(x_train)
    x_test = np.linspace(0, INTERVAL, 100).reshape(-1, 1)
    opt_lib = OptimizationLibrary(offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)
    mu_star, var_star = opt_lib.predict_and_plot(x_train, y_train, x_test, opt_lib.kernel, opt_lib.f)
    for iteration in range(n_iterations):
        EI = opt_lib.expected_improvement(x_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
        x_next = x_test[np.argmax(EI)]
        y_next = opt_lib.f(x_next)
        x_train = np.vstack((x_train, x_next))
        y_train = np.vstack((y_train, y_next))
        print(f"Iteration {iteration+1}: x_next = {x_next[0]}, y_next = {y_next[0]}")
        mu_star, var_star = opt_lib.predict_and_plot(x_train, y_train, x_test, opt_lib.kernel, opt_lib.f)
