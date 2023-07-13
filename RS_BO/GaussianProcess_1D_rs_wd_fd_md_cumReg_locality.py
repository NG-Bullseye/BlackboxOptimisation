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
        self.mu_star = K_star @ np.linalg.inv(K) @ self.y_train.flatten() + offsetkernel.flatten()
        self.var_star = np.diag(self.kernel_func(x_test, x_test) - K_star @ np.linalg.inv(K) @ K_star.T)
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

    def protection_func(self, x1, x2):
        return (x1 - x2) ** 2 / self.protection_width

    def offset_kernel(x_train, x_test, offset_scalar_func, offset_range, offset_scale, protection_width):
        offset_scalar_all = offset_scalar_func(x_train)
        print("Offset scalar for all train points:", offset_scalar_all)
        offset_matrix = np.zeros_like(x_test)

        for i in range(len(x_test)):
            offset_matrix[i] = offset_scalar_func(x_train, x_test[i], offset_scalar_func, offset_range, offset_scale,
                                               protection_width).sum()

        return offset_matrix

    def _perturbation(self, x, mu, width, a, y_offset):
        perturbations = np.zeros_like(x)
        for i, mu_i in enumerate(mu.flatten()):
            # Calculate perturbation for this training point
            perturbation = a[i] * ((x - mu_i) / width) * np.exp(-((x - mu_i) / width) ** 2) + y_offset
            perturbations += perturbation
        return perturbations

    def optimize(self, x_test, f_discrete, n_iterations, xi=0.01):
        best_y = np.max(self.y_train)  # The best function value found so far
        mu_star_final = None  # To store the final mu_star array
        quant_x_test_mu_star = []  # New list to store (x_test quantized, mu_star) pairs
        cumulative_regret = 0  # The cumulative regret

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
            # Update the best function value and the cumulative regret
            if y_next > best_y:
                best_y = y_next
            regret = best_y - y_next
            cumulative_regret += regret
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