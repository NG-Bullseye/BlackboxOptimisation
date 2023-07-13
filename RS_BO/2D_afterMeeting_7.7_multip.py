import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_results(x1_test, x2_test, mu_star, X_train, Y_train, f, PERT_WIDTH, PERT_SCALE, INTERVAL):
    X, Y = np.meshgrid(x1_test, x2_test)
    Z_true = f_discrete(np.dstack([X, Y]).reshape(-1, 2)).reshape(X.shape)
    Z_pred = mu_star.reshape(100, 100)

    mu = X_train +  gradient_vector(X_train.reshape(1, -1))
    width = PERT_WIDTH
    a = PERT_SCALE
    y_offset = 0
    Z_perturbation = perturbation(np.dstack([X, Y]).reshape(-1, 2), mu, width, a).reshape(X.shape)


    # calculate gradients at training points
    gradients = np.array([gradient_vector(X_train) for i in range(X_train.shape[0])])

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
    # add quiver (gradient vector plot) to the plot
    fig.add_trace(
        go.Cone(x=X_train[:, 0], y=X_train[:, 1], z=Y_train,
                u=gradients[:, 0], v=gradients[:, 1], w=np.zeros_like(gradients[:, 0]),
                # w is set to zero, since we have a 2D function
                sizemode='scaled', sizeref=0.2, anchor='tail'),
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
    fig.add_trace(
        go.Cone(x=X_train[:, 0], y=X_train[:, 1], z=np.zeros_like(Y_train),
                u=gradients[:, 0], v=gradients[:, 1], w=np.zeros_like(gradients[:, 0]),
                sizemode='scaled', sizeref=0.2, anchor='tail'),
        row=1, col=3)

    fig.update_layout(height=800, width=1800,
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
                                 aspectratio=dict(x=1, y=1, z=0.7),
                                 camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))),
                      scene2=dict(zaxis=dict(range=[-1, 5])))

    fig.show()


def expected_improvement(X, mu, sigma, xi=0.01):
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf_discrete(Z) + sigma * norm.pdf_discrete(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def kernel(a, b, l=1.0):
    sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-.5 * (1 / l) * sqdist)

def f_discrete(x):
    return (np.sin(x[:, 0]) + np.sin(x[:, 1])) / 5+0.5

def gradient_vector(x,GRADIENT_LENGTH=5):
    x1 = -np.cos(x[0][0]) / 5
    x2 = -np.cos(x[0][1]) / 5
    gradient = np.array([x1, x2])*GRADIENT_LENGTH
    gradient = np.array([1, 1])
    return gradient


def perturbation(x, mu, width, a):#
    print("befor Shape of mu in perturbation: ", mu.shape)
    #mu = mu.flatten()   change mu's shape to (2,)
    print("Shape of mu in perturbation: ", mu.shape)
    print(mu)
    print(mu)
    #mu = np.array([6, 7])
    print(mu)
    perturbation = a * np.exp(-np.sum(((x - mu) / width) ** 2, axis=-1))
    print("Shape of perturbation result: ", perturbation.shape)
    return perturbation


def offset_function(x_train, x_test, gradient_vector_func, PERT_WIDTH, PERT_SCALE):
    # Calculate all offsets for all training points
    print("hheheo")
    print([xt +gradient_vector_func(xt.reshape(1, -1))for xt in x_train])
    offsets = np.array([
        perturbation(x_test, xt + gradient_vector_func(xt.reshape(1, -1)), PERT_WIDTH, PERT_SCALE)
        for xt in x_train
    ])
    print("Shape of offsets in offset_function: ", offsets.shape)
    # Take the mean of the offsets
    return offsets.mean(axis=0)

def offset_kernel(X_train, X_test, gradient_vector_func, PERT_WIDTH, PERT_SCALE):
    offset_matrix = offset_function(X_train, X_test, gradient_vector_func, PERT_WIDTH, PERT_SCALE)
    return offset_matrix.ravel()  # use ravel() instead of flatten() here


def predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=3.0, PERT_SCALE=1.0, USE_OFFSET=True):
    K = kernel(X_train, X_train)
    K_star = kernel(X_test, X_train)
    print("\nShape of K:", K.shape, "First few elements:", K[:3, :3])
    print("Shape of K_star:", K_star.shape, "First few elements:", K_star[:3, :3])
    if USE_OFFSET:
        offsetkernel = offset_kernel(X_train, X_test, gradient_vector, PERT_WIDTH, PERT_SCALE)
        print("Shape of offsetkernel before flattening: ", offsetkernel.shape)
        print("Shape of K_star @ np.linalg.inv(K) @ Y_train.flatten(): ",
              (K_star @ np.linalg.inv(K) @ Y_train.flatten()).shape)

        mu_star = K_star @ np.linalg.inv(K) @ Y_train.flatten() * offsetkernel.flatten()
    else:
        mu_star = K_star @ np.linalg.inv(K) @ Y_train.flatten()
    var_star = np.diag(kernel(X_test, X_test) - K_star @ np.linalg.inv(K) @ K_star.T)
    print("Shape of mu_star:", mu_star.shape, "First few elements:", mu_star[:3])
    print("Shape of var_star:", var_star.shape, "First few elements:", var_star[:3])

    return mu_star, var_star

if __name__ == '__main__':
    PERT_WIDTH=1
    PERT_SCALE=0.3
    n_iterations = 0
    np.random.seed(42)
    INTERVAL=10

    X_train = np.array([[5,6]])#np.random.uniform(0, INTERVAL, size=(1, 2))
    Y_train = f_discrete(X_train)

    x1_test = np.linspace(0, INTERVAL, 100)
    x2_test = np.linspace(0, INTERVAL, 100)
    X_test = np.dstack(np.meshgrid(x1_test, x2_test)).reshape(-1, 2)

    # predict before the loop
    mu_star, var_star = predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=PERT_WIDTH, PERT_SCALE=PERT_SCALE)
    plot_results(x1_test, x2_test, mu_star, X_train, Y_train, f_discrete, PERT_WIDTH, PERT_SCALE,INTERVAL=INTERVAL)
    #plot_gradient_at_train_points(X_train, f, ax=None)


    for iteration in range(n_iterations):
        EI = expected_improvement(X_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
        X_next = X_test[np.argmax(EI)]
        Y_next = f_discrete(X_next[None, :])
        X_train = np.vstack((X_train, X_next))
        Y_train = np.hstack((Y_train, Y_next))

        # predict after each iteration
        mu_star, var_star = predict(X_train, Y_train, X_test, kernel, PERT_WIDTH=PERT_WIDTH, PERT_SCALE=PERT_SCALE)
