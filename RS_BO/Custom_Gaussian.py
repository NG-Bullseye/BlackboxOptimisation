import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import matplotlib.pyplot as plt


class Optimization:

    def __init__(self, get_rec_scalar,quantization_factor=1, offset_range=5, offset_scale=0.1,
                 kernel_scale=5, protection_width=1, n_iterations=3,callback_plotting=None,plotting=False,deactivate_rec_scalar=False):
        self.iteration = 0
        self.get_rec_scalar=get_rec_scalar
        self.n_iterations = n_iterations
        self.QUANTIZATION_FACTOR = quantization_factor
        self.OFFSET_RANGE = offset_range
        self.OFFSET_SCALE = offset_scale
        self.KERNEL_SCALE = kernel_scale
        self.PROTECTION_WIDTH = protection_width
        self.regrets = []
        self.plotting  = plotting
        self.callback_plotting=callback_plotting
        self.deactivate_rec_scalar=deactivate_rec_scalar

    def expected_improvement(self, X, mu, sigma, xi=0.01):
        mu_sample_opt = np.max(mu)
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    def kernel(self, a, b):
        return np.exp(-0.1 * ((a - b) ** 2 / self.KERNEL_SCALE ** 2))



    def perturbation(self,x_train, x, mu, width, a, y_offset):
        return (a * ((x - mu) / width) * np.exp(-((x - mu) / width) ** 2) + y_offset )

    def protection_function(self, x, mu, protection_width):
        return 1 - (1 - np.exp(-(x - mu) ** 2 / (2 * protection_width ** 2)))

    def exponential_smoothing(self,arr, alpha):
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def offset_vector(self, x_train, x_test, offset_scalar_func, offset_range, offset_scale, protection_width):
        nearest_train_point = x_train[np.argmin(np.abs(x_test[:, np.newaxis] - x_train), axis=1)]
        mu = nearest_train_point
        scalar_mat=offset_scalar_func(mu)
        if False:
            offset_range_offset = 0#offset_range+1
            offset_scale_offset = 0#offset_scale+1
            return 1
        else:
            if self.n_iterations!=0:
                offset_range_offset = scalar_mat *(0.01121)+0. #*(self.n_iterations/(self.n_iterations-self.iteration))+0.15#offset_range * scalar_mat 0.003+0.003
                offset_scale_offset = abs( scalar_mat) *(0.00015251) +0.#*(self.n_iterations/(self.n_iterations-self.iteration))+0.015#-offset_scale * scalar_mat 0.05+0.1
            else:
                offset_range_offset = scalar_mat *(0.11121)
                offset_scale_offset = abs(scalar_mat) * (0.01251)
        protection_term = self.protection_function(x_test, mu, protection_width=10)
        return self.perturbation(x_train,x_test, mu, offset_range_offset, offset_scale_offset, 0) #* protection_term

    def predict(self, x_train, y_train, x_test, kernel, x_discrete, offset_range=None, offset_scale=None,
                protection_width=None):
        # Convert x_train to a NumPy array if it's not already one
        x_train = np.array(x_train)

        # Add a new axis to x_train so that it becomes a column vector
        x_train_column = x_train[:, np.newaxis]

        # Compute the kernel
        K = kernel(x_train_column, x_train)
        K_star = kernel(x_test[:, np.newaxis], x_train)
        offset_kernel = self.offset_vector(x_train, x_test, self.get_rec_scalar,
                                           offset_range=offset_range or self.OFFSET_RANGE,
                                           offset_scale=offset_scale or self.OFFSET_SCALE,
                                           protection_width=protection_width or self.PROTECTION_WIDTH)
        alpha = 0.4  # Smoothing factor, 0 < alpha <= 1
        smoothed_kernel = self.exponential_smoothing(offset_kernel, alpha)
        jitter = 1e-8
        K += np.eye(K.shape[0]) * jitter
        if self.deactivate_rec_scalar:
            mu_star = K_star @ np.linalg.inv(K) @ y_train
            var_star = (self.kernel(x_test, x_test) - np.einsum('ij,ij->i', K_star @ np.linalg.inv(K), K_star))
        else:
            mu_star = K_star  @ np.linalg.inv(K) @ y_train + smoothed_kernel
            var_star = (self.kernel(x_test, x_test)+ smoothed_kernel*0.5- np.einsum('ij,ij->i', K_star @ np.linalg.inv(K), K_star))
        return mu_star, var_star

    def optimize(self, x_train, y_train, x_test, f_discrete,x_discrete):
        mu_star, var_star = self.predict(x_train, y_train, x_test, self.kernel,x_discrete)

        var = f_discrete(x_test)
        optimal_value = np.max(var)  # Optimal value in the domain
        regret = abs(optimal_value - y_train)  # Regret for this step
        self.regrets.append(regret)
        if self.plotting:
            self.callback_plotting(x_test, x_train, y_train, mu_star, var_star)
        for iteration in range(self.n_iterations):
            self.iteration=iteration
            EI = self.expected_improvement(x_test, mu_star, var_star, xi=0.1)
            v1 = np.argmax(EI)
            v2 = x_test[v1]
            x_next = x_discrete(v2)
            y_next = f_discrete(x_next)
            regret = abs(optimal_value - y_next)  # Regret for this step
            self.regrets.append(regret)  # Add it to the regret history
            x_train = np.append(x_train, x_next)
            y_train = np.append(y_train, y_next)
            #print(f"Iteration {iteration + 1}: x_next = {x_next}, y_next = {y_next}, regret = {regret}")
            mu_star, var_star = self.predict(x_train, y_train, x_test, self.kernel, x_discrete)
            #smoothing
            #var_star = gaussian_filter1d(var_star, sigma=1.0)
            #mu_star = gaussian_filter1d(mu_star, sigma=1.0)
            if self.plotting:
                self.callback_plotting(x_test,x_train,y_train, mu_star, var_star)
        return x_train, y_train, mu_star, var_star

    def get_cumulative_regret(self):
        return np.sum(self.regrets)