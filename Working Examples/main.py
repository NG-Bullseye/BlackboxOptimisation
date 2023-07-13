import numpy as np
from GaussianProcess_1D_rs_wd_fd_md import GaussianProcess
import matplotlib.pyplot as plt
from datetime import datetime

from RS_BO.Benchmark import Benchmark, Optimization, GridSearch, RandomSearch


def kernel(a, b, l=1.0):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist / l**2)

def offset_scalar(x):
    return np.cos(x)/2

QUANTIZATION_FACTOR = 1
def x_discrete(x):
    return np.round(x / QUANTIZATION_FACTOR) * QUANTIZATION_FACTOR
def f_discrete(x):
    return (np.sin(x_discrete(x)) + 1) / 2
# Add function call counter to f_discrete
def f_discrete_counter(x):
    f_discrete_counter.counter += 1
    return f_discrete(x)
f_discrete_counter.counter = 0

def plot_gp(gp, x_test, f_discrete):
    mu_star, var_star = gp.mu_star, gp.var_star
    x_train, y_train = gp.x_train, gp.y_train

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
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.title("Offset - Created at: " + current_time)
    plt.show()
def benchmark(INTERVAL,ITERATIONS):
    benchmark_classes = [Optimization, GridSearch, RandomSearch]
    for BenchmarkClass in benchmark_classes:
        if BenchmarkClass == Optimization:
            benchmark = BenchmarkClass(f_discrete, [(0, INTERVAL)], 0,ITERATIONS)
        else:
            benchmark = BenchmarkClass(f_discrete, [(0, INTERVAL)], ITERATIONS)

        optimal_x, optimal_fx, n_eval, time_taken = benchmark.run()

        print(f"\n{BenchmarkClass.__name__} Results:")
        print(f"The optimal value of x is {optimal_x}")
        print(f"The optimal value of f(x) is {optimal_fx}")
        print(f"Time taken: {time_taken}")

if __name__ == '__main__':
    ITERATIONS=5
    INTERVAL=10
    AMOUNT_OF_Initial_guesses=2
    np.random.seed(42)
    benchmark(INTERVAL,ITERATIONS)
    x_train = x_discrete(np.random.uniform(0, INTERVAL, AMOUNT_OF_Initial_guesses).reshape(-1, 1))
    y_train = f_discrete(x_train)
    #x_test = np.arange(0, INTERVAL + QUANTIZATION_FACTOR, QUANTIZATION_FACTOR).reshape(-1, 1)

    x_test = np.linspace(0, INTERVAL, 100).reshape(-1, 1)

    gp = GaussianProcess(kernel, offset_scalar, offset_range=1, offset_scale=1, random_seed=42, quantization_factor=QUANTIZATION_FACTOR,INTERVAL=INTERVAL)
    gp.train(x_train, y_train)
    gp.optimize(x_test, f_discrete_counter, ITERATIONS)
    mu_star, var_star = gp.predict(x_test)
    plot_gp(gp, x_test, f_discrete)



