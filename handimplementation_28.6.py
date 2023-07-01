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
    return np.exp(-0.5 * ((a - b)**2 / l**2))

def f(x):
    return (np.sin(x) + 1) / 2





def predict_and_plot(x_train, y_train, x_test, kernel, f):
    K = np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K[i, j] = kernel(x_train[i], x_train[j])
    K_star = np.zeros((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            K_star[i, j] = kernel(x_test[i], x_train[j])
    mu_star = K_star @ np.linalg.inv(K) @ y_train.flatten()
    var_star = np.zeros(len(x_test))
    for i in range(len(x_test)):
        var_star[i] = kernel(x_test[i], x_test[i]) - K_star[i] @ np.linalg.inv(K) @ K_star[i].T

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, f(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill(np.concatenate([x_test, x_test[::-1]]),
             np.concatenate([mu_star - 1.9600 * var_star,
                             (mu_star + 1.9600 * var_star)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("No offset")
    plt.show()

    return mu_star, var_star

n_iterations = 0
np.random.seed(42)
x_train = np.random.uniform(0, 10, 1).reshape(-1, 1)
y_train = f(x_train)
x_test = np.linspace(0, 10, 1000).reshape(-1, 1)

# predict and plot before the loop
mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f)

for iteration in range(n_iterations):
    EI = expected_improvement(x_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
    x_next = x_test[np.argmax(EI)]
    y_next = f(x_next)
    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))
    print(f"Iteration {iteration+1}: x_next = {x_next[0]}, y_next = {y_next[0]}")

    # predict and plot after each iteration
    mu_star, var_star = predict_and_plot(x_train, y_train, x_test, kernel, f)

