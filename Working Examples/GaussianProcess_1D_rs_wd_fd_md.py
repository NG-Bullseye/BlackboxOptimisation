import numpy as np
from scipy.stats import norm

class GaussianProcess:
    def __init__(self, kernel_func, offset_scalar_func, offset_range, offset_scale, random_seed,quantization_factor,INTERVAL):
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

    def x_discrete(self, x):
        return np.round(x / self.quantization_factor) * self.quantization_factor

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        x_test = self.x_discrete(x_test)  # Discretize x_test
        K = self.kernel_func(self.x_train, self.x_train)
        #K += 1e-6 * np.eye(K.shape[0])  # Add small noise to the diagonal
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
        #print("Offset scalar for all train points:", offset_scalar_all)

        offset_matrix = offsets.sum(axis=1)
        return offset_matrix

    def _perturbation(self, x, mu, width, a, y_offset):
        return a * ((x[:, np.newaxis] - mu[np.newaxis, :]) / width) * np.exp(-((x[:, np.newaxis] - mu[np.newaxis, :]) / width) ** 2) + y_offset

    def optimize(self, x_test, f_discrete, n_iterations, xi=0.01):
        mu_star_final = None  # To store the final mu_star array
        quant_x_test_mu_star = []  # New list to store (x_test quantized, mu_star) pairs
        for iteration in range(n_iterations):
            # Generate random points in the domain
            n_random_points =round(self.INTERVAL/self.quantization_factor) # You can adjust this value
            x_random = np.random.uniform(x_test.min(), x_test.max(), n_random_points).reshape(-1, 1)
            x_random = self.x_discrete(x_random)  # discretize x_random

            # Predict for the randomly generated points
            self.predict(x_random)
            EI_random = self.expected_improvement(x_random, xi=xi)

            # Choose the next point with the highest EI
            x_next =  self.x_discrete(x_random[np.argmax(EI_random)])

            y_next = f_discrete(x_next)

            # Add the new point to the training set
            self.x_train = np.vstack((self.x_train, x_next))
            self.y_train = np.vstack((self.y_train, y_next))

            print(f"Iteration {iteration + 1} done : x_next = {x_next[0]}, y_next = {y_next[0]}")

            # Update the objective function values array
            index = int(x_next[0])  # Convert x_next to an integer index

            if iteration == n_iterations - 1:
                # Store the final mu_star array
                mu_star_final = self.mu_star
                unique_x_test = np.unique(self.x_discrete(x_test).flatten())
                quant_x_test_mu_star = list(zip(unique_x_test, mu_star_final))

            # Predict with the updated training set
            self.predict(x_test)
        print("Objective function values (X_train Y_train):")
        print(np.hstack((self.x_train, self.y_train)))
        print("Quantized x_test and mu_star values:")
        print(quant_x_test_mu_star)