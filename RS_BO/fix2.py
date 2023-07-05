import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_perturbation(x1_test, x2_test, mu, PERT_WIDTH, PERT_SCALE):
    a = PERT_SCALE * offset_scalar(mu.reshape(1, -1))
    y_offset = 0

    X, Y = np.meshgrid(x1_test, x2_test)
    Z = perturbation(np.dstack([X, Y]).reshape(-1, 2), mu, PERT_WIDTH, a, y_offset).reshape(X.shape)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='viridis')])
    fig.update_layout(title=f'perturbation at (5,5) with WIDTH={PERT_WIDTH}, SCALE={PERT_SCALE}',
                      autosize=False, width=500, height=500,
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))
    fig.show()


def plot_results(x1_test, x2_test, mu_star, X_train, Y_train, f, PERT_WIDTH, PERT_SCALE, INTERVAL):
    X, Y = np.meshgrid(x1_test, x2_test)
    Z_true = f(np.dstack([X, Y]).reshape(-1, 2)).reshape(X.shape)
    Z_pred = mu_star.reshape(100, 100)

    width = PERT_WIDTH
    print(X_train[0,:])
    a = PERT_SCALE * offset_scalar(np.array(X_train[0,:]).reshape(1, -1))
    y_offset = 0
    Z_perturbation = perturbation(np.dstack([X, Y]).reshape(-1, 2), np.array(X_train[0,:]), width, a,
                                  y_offset).reshape(X.shape)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("True Function", "Model's Prediction", f'Perturbation WIDTH={PERT_WIDTH}, SCALE={PERT_SCALE}'),
                        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_true, colorscale='viridis'),
        row=1, col=1)
    fig.add_trace(
        go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=Y_train,
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=1)

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_pred, colorscale='viridis'),
        row=1, col=2)
    fig.add_trace(
        go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=Y_train,
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=2)

    fig.add_trace(
        go.Surface(z=Z_perturbation, x=X, y=Y, colorscale='viridis'),
        row=1, col=3)
    fig.add_trace(
        go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=Y_train,
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=3)

    fig.update_layout(height=800, width=1800,
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
                                 aspectratio=dict(x=1, y=1, z=0.7),
                                 camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))),
                      scene2=dict(zaxis=dict(range=[-1, 1])))

    fig.show()


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
    sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-.5 * (1 / l) * sqdist)

def f(x):
    return (np.sin(x[:, 0]) + np.sin(x[:, 1])) / 5+0.5

def offset_scalar(x):
    return (np.cos(x[:, 0]) + np.cos(x[:, 1])) / 5

def perturbation(x, mu, width, a, y_offset):
    perturbation = a * np.sum(((x - mu) / width) ** 2, axis=-1) * np.exp(-np.sum((x - mu) / width, axis=-1) ** 2) + y_offset
    return perturbation

def offset_function(x_train, x_test, offset_scalar_func, PERT_WIDTH, PERT_SCALE):
    offsets = np.zeros((len(x_test),))
    y_offset = 0
    for i, x in enumerate(x_test):
        for train_point in x_train:
            mu = train_point.reshape(1, -1) # reshape the mu into an array of points
            width = PERT_WIDTH
            a = PERT_SCALE * offset_scalar_func(mu)
            offsets[i] += perturbation(x, mu, width, a, y_offset)
    return np.sum(offsets)

def offset_kernel(X_train, X_test, offset_scalar_func, PERT_WIDTH, PERT_SCALE):
    offset_matrix = np.zeros((len(X_test),))

    for i in range(len(X_test)):
        offset_matrix[i] = offset_function(X_train, X_test[i], offset_scalar_func, PERT_WIDTH, PERT_SCALE)
    return offset_matrix

def predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=3.0, PERT_SCALE=1.0):
    K = kernel(X_train, X_train)
    K_star = kernel(X_test, X_train)
    print("\nShape of K:", K.shape, "First few elements:", K[:3, :3])
    print("Shape of K_star:", K_star.shape, "First few elements:", K_star[:3, :3])

    offsetkernel = offset_kernel(X_train, X_test, offset_scalar, PERT_WIDTH, PERT_SCALE)
    print("Shape of offsetkernel:", offsetkernel.shape, "First few elements:", offsetkernel[:3])

    mu_star = K_star @ np.linalg.inv(K) @ Y_train.flatten() + offsetkernel.flatten()
    var_star = np.diag(kernel(X_test, X_test) - K_star @ np.linalg.inv(K) @ K_star.T)
    print("Shape of mu_star:", mu_star.shape, "First few elements:", mu_star[:3])
    print("Shape of var_star:", var_star.shape, "First few elements:", var_star[:3])

    return mu_star, var_star

if __name__ == '__main__':
    PERT_WIDTH=0.2
    PERT_SCALE=0.005
    n_iterations = 0
    np.random.seed(42)
    INTERVAL=10

    X_train = np.random.uniform(0, INTERVAL, size=(1, 2))
    Y_train = f(X_train)

    x1_test = np.linspace(0, INTERVAL, 100)
    x2_test = np.linspace(0, INTERVAL, 100)
    X_test = np.dstack(np.meshgrid(x1_test, x2_test)).reshape(-1, 2)

    # predict before the loop
    mu_star, var_star = predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=PERT_WIDTH, PERT_SCALE=PERT_SCALE)
    plot_results(x1_test, x2_test, mu_star, X_train, Y_train, f, PERT_WIDTH, PERT_SCALE,INTERVAL=INTERVAL)
