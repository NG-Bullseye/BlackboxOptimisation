import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

QUANTIZATION_FACTOR = 2
OFFSET_RANGE = 1
OFFSET_SCALE = 1
n_iterations = 2
np.random.seed(42)
INTERVAL = 10

def expected_improvement(X, mu, sigma, xi=0.01):
    imp = mu.reshape(-1, 1) - np.max(mu) - xi
    Z = imp / sigma.reshape(-1, 1)
    ei = imp * norm.cdf(Z) + sigma.reshape(-1, 1) * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

def kernel(a, b, l=1.0):
    return np.exp(-0.5 * ((a - b)**2 / l**2))

def f(x):
    return (np.sin(x) + 1) / 2

def x_discrete(x):
    return np.round(x / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR

def f_discrete(x):
    return f(x)

def offset_scalar(x):
    return np.cos(x)/2

def offset_function(x_train, x_test, offset_scalar_func, offset_range, offset_scale):
    a = offset_scale * offset_scalar_func(x_train)
    return sum([a * ((x - mu) / offset_range) * np.exp(-((x - mu) / offset_range) ** 2) for mu in x_train])

def compute_K(x_train, kernel_func):
    return np.array([[kernel_func(x_i.item(), x_j.item()) for x_j in x_train] for x_i in x_train])

def compute_K_star(x_train, x_test, kernel_func):
    return np.array([[kernel_func(x_i.item(), x_j.item()) for x_j in x_train] for x_i in x_test])

def compute_var_star(x_test, K_star, K, kernel_func):
    return np.array([kernel_func(x_i.item(), x_i.item()) - K_star_i @ np.linalg.inv(K) @ K_star_i.T for x_i, K_star_i in zip(x_test, K_star)])

def predict_and_plot(x_train, y_train, x_test, kernel, f, offset_range=3.0, offset_scale=1.0):
    K = compute_K(x_train, kernel)
    K_star = compute_K_star(x_train, x_test, kernel)
    offsetkernel = offset_function(x_train, x_test, offset_scalar, offset_range, offset_scale)
    mu_star = K_star @ np.linalg.inv(K) @ y_train.flatten() + offsetkernel.flatten()
    var_star = compute_var_star(x_test, K_star, K, kernel)

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill(np.concatenate([x_test, x_test[::-1]]),
             np.concatenate([mu_star - 1.9600 * var_star,
                             (mu_star + 1.9600 * var_star)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("Offset")
    plt.show()

    return mu_star, var_star

x_train = x_discrete(np.random.uniform(0, INTERVAL, 2).reshape(-1, 1))
y_train = f_discrete(x_train)
x_test = np.linspace(0, INTERVAL, 100).reshape(-1, 1)

mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)

for iteration in range(n_iterations):
    EI = expected_improvement(x_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
    x_next = x_discrete(x_test[np.argmax(EI)])
    y_next = f_discrete(x_next)
    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))

    mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)
