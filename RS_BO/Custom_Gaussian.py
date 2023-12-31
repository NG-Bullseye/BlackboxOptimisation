import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Optimization:

    def __init__(self, n_iterations=3, quantization_factor=1, offset_range=5, offset_scale=0.1,
                 kernel_scale=5, protection_width=1):
        self.n_iterations = n_iterations
        self.QUANTIZATION_FACTOR = quantization_factor
        self.OFFSET_RANGE = offset_range
        self.OFFSET_SCALE = offset_scale
        self.KERNEL_SCALE = kernel_scale
        self.PROTECTION_WIDTH = protection_width
        self.regrets = []

    def expected_improvement(self, X, mu, sigma, xi=0.01):
        mu_sample_opt = np.max(mu)
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    def kernel(self, a, b):
        return np.exp(-0.5 * ((a - b) ** 2 / self.KERNEL_SCALE ** 2))

    @staticmethod
    def f(x):
        return (np.sin(0.1 * x) + 1) / 2

    def x_discrete(self, x):
        return np.round(x / self.QUANTIZATION_FACTOR) * self.QUANTIZATION_FACTOR

    def f_discrete(self, x):
        return self.f(self.x_discrete(x))

    @staticmethod
    def offset_scalar(x):
        return np.cos(x) / 2

    def perturbation(self, x, mu, width, a, y_offset):
        return a * ((x - mu) / width) * np.exp(-((x - mu) / width) ** 2) + y_offset

    def protection_function(self, x, mu, protection_width):
        return 1 - (1 - np.exp(-(x - mu) ** 2 / (2 * protection_width ** 2)))

    def offset_vector(self, x_train, x_test, offset_scalar_func, offset_range, offset_scale, protection_width):
        nearest_train_point = x_train[np.argmin(np.abs(x_test[:, np.newaxis] - x_train), axis=1)]
        mu = nearest_train_point
        a = offset_scale * offset_scalar_func(mu)
        protection_term = self.protection_function(x_test, mu, protection_width)
        return self.perturbation(x_test, mu, offset_range, a, 0) * protection_term

    def predict(self, x_train, y_train, x_test, kernel, f, offset_range=None, offset_scale=None, protection_width=None):
        x_test = self.x_discrete(x_test)
        K = kernel(x_train[:, np.newaxis], x_train)
        K_star = kernel(x_test[:, np.newaxis], x_train)
        offset_kernel = self.offset_vector(x_train, x_test, self.offset_scalar,
                                           offset_range=offset_range or self.OFFSET_RANGE,
                                           offset_scale=offset_scale or self.OFFSET_SCALE,
                                           protection_width=protection_width or self.PROTECTION_WIDTH)
        jitter = 1e-6
        K += np.eye(K.shape[0]) * jitter
        mu_star = K_star @ np.linalg.inv(K) @ y_train + offset_kernel
        var_star = self.kernel(x_test, x_test) - np.einsum('ij,ij->i', K_star @ np.linalg.inv(K), K_star)
        return mu_star, var_star

    def optimize(self, x_train, y_train, x_test):
        mu_star, var_star = self.predict(x_train, y_train, x_test, self.kernel, self.f)
        for iteration in range(self.n_iterations):
            EI = self.expected_improvement(x_test, mu_star, var_star, xi=0.01)
            x_next = self.x_discrete(x_test[np.argmax(EI)])
            y_next = self.f_discrete(x_next)
            x_train = np.append(x_train, x_next)
            y_train = np.append(y_train, y_next)
            print(f"Iteration {iteration + 1}: x_next = {x_next}, y_next = {y_next}")
            mu_star, var_star = self.predict(x_train, y_train, x_test, self.kernel, self.f)
        return x_train, y_train, mu_star, var_star
