import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

QUANTIZATION_FACTOR, OFFSET_RANGE, OFFSET_SCALE, KERNEL_SCALE, PROTECTION_WIDTH = 1, 5, 0.1, 5, 1

def expected_improvement(X, mu, sigma, xi=0.01):
    Z = (mu - np.max(mu) - xi) / sigma
    ei = (mu - np.max(mu) - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

def kernel(a, b):
    return np.exp(-0.5 * ((a - b)**2 / KERNEL_SCALE**2))

def predict(x_train, y_train, x_test):
    x_test = np.round(x_test / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR
    K = kernel(x_train[:, np.newaxis], x_train)
    K_star = kernel(x_test[:, np.newaxis], x_train)
    nearest_train_point = x_train[np.argmin(np.abs(x_test[:, np.newaxis] - x_train), axis=1)]
    mu = nearest_train_point
    a = OFFSET_SCALE * np.cos(mu) / 2
    protection_term = 1 - (1 - np.exp(-(x_test - mu) ** 2 / (2 * PROTECTION_WIDTH ** 2)))
    offset_kernel = a * ((x_test - mu) / OFFSET_RANGE) * np.exp(-((x_test - mu) / OFFSET_RANGE) ** 2) * protection_term
    jitter = 1e-6
    K += np.eye(K.shape[0]) * jitter
    mu_star = K_star @ np.linalg.inv(K) @ y_train + offset_kernel
    var_star = kernel(x_test, x_test) - np.einsum('ij,ij->i', K_star @ np.linalg.inv(K), K_star)
    return mu_star, var_star

if __name__ == '__main__':
    np.random.seed(42)
    x_train = np.round(np.random.uniform(0, 100, 1) / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR
    y_train = (np.sin(0.1*x_train) + 1) / 2
    x_test = np.linspace(0, 100, 100)
    mu_star, var_star = predict(x_train, y_train, x_test)
    for _ in range(3):
        EI = expected_improvement(x_test, mu_star, var_star, xi=0.01)
        x_next = np.round(x_test[np.argmax(EI)] / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR
        y_next = (np.sin(0.1*x_next) + 1) / 2
        x_train, y_train = np.append(x_train,x_next), np.append(y_train, y_next)
        mu_star, var_star = predict(x_train, y_train, x_test)
    plt.figure(figsize=(12, 8))
    plt.plot(x_test, (np.sin(0.1*x_test) + 1) / 2, 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill_between(x_test, mu_star - 1.9600 * var_star, mu_star + 1.9600 * var_star, color='b', alpha=.5, label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("Offset")
    plt.show()
