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

def kernel(a, b):
    return np.exp(-0.5 * ((a - b)**2 / KERNEL_SCALE**2))

def f(x):
    return (np.sin(0.1*x) + 1) / 2

QUANTIZATION_FACTOR = 1
def x_discrete(x):
    #return x
    return np.round(np.round(x / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR)

def f_discrete(x):
    return f(x_discrete(x))

def offset_scalar(x):
    return np.cos(x)/2


def perturbation(x, mu, width, a, y_offset):
    perturbation = a * ((x - mu) / width) * np.exp(-((x - mu) / width) ** 2) + y_offset
    return perturbation

def protection_function(x, mu, protection_width):
    return 1 - (1 - np.exp(-(x - mu) ** 2 / (2 * protection_width ** 2)))

def offset_vector(x_train, x_test, offset_scalar_func, offset_range, offset_scale, protection_width):
    offsets = np.zeros_like(x_test)

    for i, x in enumerate(x_test):
        # Find the nearest training point
        nearest_train_point = x_train[np.argmin(np.abs(x - x_train))]
        mu = nearest_train_point
        width = offset_range
        a = offset_scale * offset_scalar_func(mu)
        y_offset = 0
        # Use only the protection term from the nearest training point
        protection_term = protection_function(x, mu, protection_width)
        offsets[i] = perturbation(x, mu, width, a, y_offset) * protection_term

    return offsets

def predict(x_train, y_train, x_test, kernel, f, offset_range=3.0, offset_scale=1.0,protection_width=1):
    x_test = x_discrete(x_test)  # Discretize x_test
    K = np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K[i, j] = kernel(x_train[i], x_train[j])
    K_star = np.zeros((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            K_star[i, j] = kernel(x_test[i], x_train[j])
    offsetkernel = offset_vector(x_train, x_test, offset_scalar, offset_range=offset_range, offset_scale=offset_scale,protection_width=protection_width)
    jitter = 1e-6  # Small constant. You may adjust this value as per your needs.
    K += np.eye(K.shape[0]) * jitter
    mu_star = K_star @ np.linalg.inv(K) @ y_train.flatten() + offsetkernel.flatten()
    #print("mu_star:", mu_star)
    var_star = np.zeros(len(x_test))
    for i in range(len(x_test)):
        var_star[i] = kernel(x_test[i], x_test[i]) - K_star[i] @ np.linalg.inv(K) @ K_star[i].T
    return mu_star, var_star

OFFSET_RANGE = 5
OFFSET_SCALE = 0.1
KERNEL_SCALE = 5
PROTECTION_WIDTH = 1
if __name__ == '__main__':
    n_iterations = 3
    np.random.seed(42)
    INTERVAL=100
    x_train = x_discrete(  np.random.uniform(0, INTERVAL, 1))#.reshape(-1, 1))
    y_train = f_discrete(x_train)

    x_test = np.linspace(0, INTERVAL, 100)#.reshape(-1, 1)

    # predict and plot before the loop
    mu_star, var_star = predict(x_train, y_train, x_test, kernel, f, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE, protection_width= PROTECTION_WIDTH)


    for iteration in range(n_iterations):
        EI = expected_improvement(x_test, mu_star.reshape(-1, 1), var_star.reshape(-1, 1), xi=0.01)
        x_next = x_discrete(x_test[np.argmax(EI)])
        y_next = f_discrete(x_next)
        x_train =np.append(x_train,x_next) #= np.vstack((x_train, x_next))
        y_train = np.append(y_train, y_next)
        print(f"Iteration {iteration+1}: x_next = {x_next}, y_next = {y_next}")

        # predict and plot after each iteration
        mu_star, var_star = predict(x_train, y_train, x_test, kernel, f, offset_range=OFFSET_RANGE, offset_scale=OFFSET_SCALE)
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