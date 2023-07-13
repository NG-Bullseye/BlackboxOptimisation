import numpy as np
from scipy.stats import norm

class GaussianProcess:
    def __init__(self, kernel_func, offset_scalar_func, offset_range, offset_scale, random_seed,quantization_factor,INTERVAL,protection_width):
        np.random.seed(random_seed)
        self.INTERVAL=INTERVAL
        self.kernel_func = kernel_func
        self.offset_scalar_func = offset_scalar_func
        self.offset_range = offset_range
        self.offset_scale = offset_scale
        self.x_train = None
        self.y_train = None
        self.mu_star = None
        self.var_star = None
        self.quantization_factor = quantization_factor
        self.protection_width=protection_width

    def x_discrete(self, x):
        return np.round(x / self.quantization_factor) * self.quantization_factor

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        x_test = self.x_discrete(x_test)  # Discretize x_test
        K = self.kernel_func(self.x_train, self.x_train)
        K_star = self.kernel_func(x_test, self.x_train)
        offsetkernel = self._offset_kernel(x_test)
        jitter = 1e-6  # Small constant. You may adjust this value as per your needs.
        K += np.eye(K.shape[0]) * jitter
        self.mu_star = K_star @ np.linalg.inv(K) @ self.y_train.flatten() + offsetkernel.flatten()
        self.var_star = np.zeros(len(x_test))
        for i in range(len(x_test)):
            self.var_star[i] = self.kernel_func(x_test[i].reshape(-1, 1), x_test[i].reshape(-1, 1)) - K_star[
                i] @ np.linalg.inv(K) @ K_star[i].T
        return self.mu_star, self.var_star

    def expected_improvement(self, x_test, xi=0.01):
        mu_star = self.mu_star.reshape(-1, 1)
        var_star = self.var_star.reshape(-1, 1)
        mu_sample_opt = np.max(mu_star)
        with np.errstate(divide='warn'):
            imp = mu_star - mu_sample_opt - xi
            Z = imp / (var_star + 1e-8)
            ei = imp * norm.cdf(Z) + var_star * norm.pdf(Z)
            ei[var_star == 0.0] = 0.0
        return ei

    def _offset_kernel(self, x_test):
        offsets = np.zeros_like(x_test)

        for i, x in enumerate(x_test):
            for train_point in self.x_train:
                mu = train_point
                width = self.offset_range
                a = self.offset_scale * self.offset_scalar_func(mu)
                y_offset = 0
                offsets[i] += self._perturbation_protection(x, mu, width, a, y_offset)

        return offsets

    def _perturbation_protection(self, x, mu, width, a, y_offset):
        protection_term = self.protection_func(x, mu)
        perturbation = self._perturbation(x, mu, width, a, y_offset)
        return perturbation * protection_term

    def protection_func(self, x, mu):
        return 1 / ((x - mu) ** 2 / self.protection_width+ 1e-8)

    def _perturbation(self, x, mu, width, a, y_offset):
        return a * ((x - mu) / width) * np.exp(-((x - mu) / width) ** 2) + y_offset

    def optimize(self, x_test, f_discrete, n_iterations, xi=0.01):
        self.predict(x_test)  # Add this line here
        for iteration in range(n_iterations):
            EI = self.expected_improvement(x_test, xi=xi)
            x_next = self.x_discrete(x_test[np.argmax(EI)])
            y_next = f_discrete(x_next)
            self.x_train = np.vstack((self.x_train, x_next))
            self.y_train = np.vstack((self.y_train, y_next))
            print(f"Iteration {iteration+1}: x_next = {x_next[0]}, y_next = {y_next[0]}")
            self.predict(x_test)

