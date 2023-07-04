import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/l**2) * sqdist)

def f(x, y):
    return (np.sin(x) + np.sin(y)) / 2

QUANTIZATION_FACTOR = 2
def xy_discrete(xy):
    return np.round(xy / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR

def offset_scalar(x):
    return np.cos(np.sum(x**2, axis=1))/2

def predict(x_train, y_train, x_test, kernel, offset_range=3.0, offset_scale=1.0):
    K = kernel(x_train, x_train)
    K_star = kernel(x_test, x_train)
    mu_star = K_star @ np.linalg.inv(K) @ y_train
    var_star = np.diag(kernel(x_test, x_test) - K_star @ np.linalg.inv(K) @ K_star.T)
    return mu_star, var_star
if __name__ == '__main__':

    OFFSET_RANGE=1
    OFFSET_SCALE=1
    n_iterations = 2
    np.random.seed(42)
    INTERVAL=10
    x_train = xy_discrete(np.random.uniform(0, INTERVAL, size=(2, 2)))
    y_train = f(x_train[:, 0], x_train[:, 1])

    x1_test = np.linspace(0, INTERVAL, 100)
    x2_test = np.linspace(0, INTERVAL, 100)
    x_test = np.array([[x1, x2] for x1 in x1_test for x2 in x2_test])

    # predict and plot before the loop
    mu_star, var_star = predict(x_train, y_train, x_test, kernel, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)

    for iteration in range(n_iterations):
        EI = expected_improvement(x_test, mu_star, var_star, xi=0.01)
        x_next = xy_discrete(x_test[np.argmax(EI)])
        y_next = f(x_next[0], x_next[1])
        x_train = np.vstack((x_train, x_next))
        y_train = np.append(y_train, y_next)
        print(f"Iteration {iteration+1}: x_next = {x_next}, y_next = {y_next}")

        # predict and plot after each iteration
        mu_star, var_star = predict(x_train, y_train, x_test, kernel, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)
