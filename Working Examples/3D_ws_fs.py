import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.metrics.pairwise import euclidean_distances

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

from scipy.spatial.distance import pdist, squareform

def kernel(a, b, l=1.0):
    sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-.5 * (1/l**2) * sqdist)


def f(x, y):
    return (np.sin(x) + np.sin(y)) / 2

def predict(x_train, y_train, x_test, kernel):
    K = kernel(x_train, x_train)
    K_star = kernel(x_test, x_train)
    mu_star = K_star @ np.linalg.inv(K) @ y_train
    var_star = np.diag(kernel(x_test, x_test) - K_star @ np.linalg.inv(K) @ K_star.T)
    return mu_star.ravel(), var_star

if __name__ == '__main__':
    n_iterations = 0
    np.random.seed(42)
    INTERVAL=10
    x_train = np.random.uniform(0, INTERVAL, size=(2, 2))
    y_train = f(x_train[:, 0], x_train[:, 1])

    x1_test = np.linspace(0, INTERVAL, 100)
    x2_test = np.linspace(0, INTERVAL, 100)
    x_test = np.dstack(np.meshgrid(x1_test, x2_test)).reshape(-1, 2)
    mu_star, var_star = predict(x_train, y_train.reshape(-1, 1), x_test, kernel)

    for iteration in range(n_iterations):
        EI = expected_improvement(mu_star, mu_star, var_star, xi=0.01)
        x_next = x_test[np.argmax(EI)]
        y_next = f(x_next[0], x_next[1])
        x_train = np.vstack((x_train, x_next))
        y_train = np.append(y_train, y_next)
        print(f"Iteration {iteration+1}: x_next = {x_next}, y_next = {y_next}")
        mu_star, var_star = predict(x_train, y_train[:, None], x_test, kernel)

    # Create a grid of points
    X, Y = np.meshgrid(x1_test, x2_test)
    Z_true = (np.sin(X) + np.sin(Y)) / 2
    Z_pred = mu_star.reshape(100, 100)

    fig = sp.make_subplots(rows=1, cols=2,
        subplot_titles=("True Function", "Model's Prediction"),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_true, colorscale='viridis'),
        row=1, col=1)

    fig.add_trace(
        go.Scatter3d(x=x_train[:, 0], y=x_train[:, 1], z=y_train,
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=1)

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_pred, colorscale='viridis'),
        row=1, col=2)

    fig.add_trace(
        go.Scatter3d(x=x_train[:, 0], y=x_train[:, 1], z=y_train,
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=2)

    fig.update_layout(height=800, width=1200,
        scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
        aspectratio=dict(x=1, y=1, z=0.7),
        camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))),
        scene2 = dict(zaxis=dict(range=[-1, 1])))
    fig.show()
