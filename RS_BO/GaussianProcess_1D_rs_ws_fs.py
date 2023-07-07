import numpy as np
from scipy.stats import norm

class GaussianProcess:
    def __init__(self, kernel_func, offset_scalar_func, offset_range, offset_scale,random_seed):
        np.random.seed(random_seed)
        self.kernel_func = kernel_func
        self.offset_scalar_func = offset_scalar_func
        self.offset_range = offset_range
        self.offset_scale = offset_scale
        self.x_train = None
        self.y_train = None
        self.mu_star = None
        self.var_star = None

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        K = self.kernel_func(self.x_train, self.x_train)
        K_star = self.kernel_func(x_test, self.x_train)
        offsetkernel = self._offset_kernel(x_test)
        self.mu_star = K_star @ np.linalg.inv(K) @ self.y_train.flatten() + offsetkernel.flatten()
        self.var_star = np.diag(self.kernel_func(x_test, x_test) - K_star @ np.linalg.inv(K) @ K_star.T)
        return self.mu_star, self.var_star

    def expected_improvement(self, x_test, xi=0.01):
        mu_star = self.mu_star.reshape(-1, 1)
        var_star = self.var_star.reshape(-1, 1)
        mu_sample_opt = np.max(mu_star)
        with np.errstate(divide='warn'):
            imp = mu_star - mu_sample_opt - xi
            Z = imp / var_star
            ei = imp * norm.cdf(Z) + var_star * norm.pdf(Z)
            ei[var_star == 0.0] = 0.0
        return ei

    def _offset_kernel(self, x_test):
        mu = self.x_train
        width = self.offset_range
        a = self.offset_scale * self.offset_scalar_func(mu)
        y_offset = 0
        offsets = self._perturbation(x_test, mu, width, a, y_offset)

        offset_scalar_all = self.offset_scalar_func(self.x_train)
        print("Offset scalar for all train points:", offset_scalar_all)

        offset_matrix = offsets.sum(axis=1)
        return offset_matrix

    def _perturbation(self, x, mu, width, a, y_offset):
        return a * ((x[:, np.newaxis] - mu[np.newaxis, :]) / width) * np.exp(-((x[:, np.newaxis] - mu[np.newaxis, :]) / width) ** 2) + y_offset

    @staticmethod
    def _x_discrete(x, quantization_factor=2):
        return np.round(x / quantization_factor) * quantization_factor
    def optimize(self, x_test, f_discrete, n_iterations, xi=0.01):
        for iteration in range(n_iterations):
            EI = self.expected_improvement(x_test, xi=xi)
            x_next = self._x_discrete(x_test[np.argmax(EI)])
            y_next = f_discrete(x_next)
            self.x_train = np.vstack((self.x_train, x_next))
            self.y_train = np.vstack((self.y_train, y_next))
            print(f"Iteration {iteration + 1}: x_next = {x_next[0]}, y_next = {y_next[0]}")

            # predict after each iteration
            self.predict(x_test)