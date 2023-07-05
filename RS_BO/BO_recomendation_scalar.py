import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

class GaussianProcessOptimizer:
    def __init__(self, function, offset_scalar_func, quantization_factor=2, n_iterations=20, interval=10, offset_range=1.0, offset_scale=1.0):
        self.f = function
        self.offset_scalar_func = offset_scalar_func
        self.QUANTIZATION_FACTOR = quantization_factor
        self.n_iterations = n_iterations
        self.INTERVAL = interval
        self.OFFSET_RANGE = offset_range
        self.OFFSET_SCALE = offset_scale
        self.X_train = self.xy_discrete(np.random.uniform(0, self.INTERVAL, size=(2, 2)))
        self.Y_train = self.f_discrete(self.X_train)
        self.kernel = self.rbf_kernel

    def xy_discrete(self, xy):
        return np.round(xy / self.QUANTIZATION_FACTOR) * self.QUANTIZATION_FACTOR

    def f_discrete(self, xy):
        return self.f(xy[:, 0], xy[:, 1])

    def rbf_kernel(self, a, b, l=1.0):
        sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-.5 * (1 / l ** 2) * sqdist)

    def predict(self, X_test):
        K = self.kernel(self.X_train, self.X_train)
        K_star = self.kernel(X_test, self.X_train)
        offsetkernel = self.offset_kernel(self.X_train, X_test)
        mu_star = K_star @ np.linalg.inv(K) @ self.Y_train[:, None] + offsetkernel.flatten()
        var_star = np.diag(self.kernel(X_test, X_test) - K_star @ np.linalg.inv(K) @ K_star.T)
        return mu_star.ravel(), var_star

    def perturbation(x, mu, width, a, y_offset):
        # Make sure x, mu and y_offset are numpy arrays
        x = np.asarray(x)
        mu = np.asarray(mu)
        y_offset = np.asarray(y_offset)

        perturbation = a * ((x - mu) / width) ** 2 * np.exp(-np.sum((x - mu) / width) ** 2) + y_offset
        return perturbation

    def offset_function(x_train, x_test, offset_scalar_func, offset_range, offset_scale):
        offsets = np.zeros_like(x_test)

        for i, x in enumerate(x_test):
            for train_point in x_train:
                mu = train_point  # The mean is the training point itself
                width = offset_range
                a = offset_scale * offset_scalar_func(mu)
                y_offset = np.zeros(2)  # Replace this with your actual 2D y_offset
                offsets[i] += perturbation(x, mu, width, a, y_offset)

        return offsets

    def offset_kernel(self, x_train, x_test):
        offset_scalar_all = self.offset_scalar_func(x_train)
        offset_matrix = np.zeros_like(x_test)

        for i in range(len(x_test)):
            offset_matrix[i] = self.offset_function(x_train, x_test[i])

        return offset_matrix
    def expected_improvement(self, mu, sigma, xi=0.01):
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def optimize(self):
        x1_test = np.linspace(0, self.INTERVAL, 100)
        x2_test = np.linspace(0, self.INTERVAL, 100)
        X_test = self.xy_discrete(np.dstack(np.meshgrid(x1_test, x2_test)).reshape(-1, 2))
        mu_star, var_star = self.predict(X_test)

        for iteration in range(self.n_iterations):
            EI = self.expected_improvement(mu_star, var_star, xi=0.01)
            X_next = X_test[np.argmax(EI)]
            Y_next = self.f_discrete(X_next[None, :])
            self.X_train = np.vstack((self.X_train, X_next))
            self.Y_train = np.append(self.Y_train, Y_next)
            mu_star, var_star = self.predict(X_test)

        self.plot_results(x1_test, x2_test, mu_star)

    def plot_results(self, x1_test, x2_test, mu_star):
        X, Y = np.meshgrid(x1_test, x2_test)
        Z_true = self.f_discrete(np.dstack([X, Y]).reshape(-1, 2)).reshape(X.shape)
        Z_pred = mu_star.reshape(100, 100)
        fig = sp.make_subplots(rows=1, cols=2,
                               subplot_titles=("True Function", "Model's Prediction"),
                               specs=[[{'type': 'surface'}, {'type': 'surface'}]])

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z_true, colorscale='viridis'),
            row=1, col=1)
        fig.add_trace(
            go.Scatter3d(x=self.X_train[:, 0], y=self.X_train[:, 1], z=self.Y_train,
                         mode='markers', marker=dict(size=4, color='red')),
            row=1, col=1)

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z_pred, colorscale='viridis'),
            row=1, col=2)

        fig.add_trace(
            go.Scatter3d(x=self.X_train[:, 0], y=self.X_train[:, 1], z=self.Y_train,
                         mode='markers', marker=dict(size=4, color='red')),
            row=1, col=2)

        fig.update_layout(height=800, width=1200,
                          scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
                                     aspectratio=dict(x=1, y=1, z=0.7),
                                     camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))),
                          scene2=dict(zaxis=dict(range=[-1, 1])))
        fig.show()


def offset_scalar(x):
    return np.cos(x) / 2
if __name__ == '__main__':
    np.random.seed(42)
    function_to_optimize = lambda x, y: (np.sin(x) + np.sin(y)) / 2
    optimizer = GaussianProcessOptimizer(function_to_optimize,offset_scalar_func=offset_scalar)
    optimizer.optimize()