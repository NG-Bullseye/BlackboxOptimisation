import numpy as np
from RS_BO.Custom_Gaussian import Optimization as opt
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional

from RS_BO.Benchmark import Benchmark, Optimization, GridSearch, RandomSearch
from RS_BO.Utility.Sim import Sim

QUANTIZATION_FACTOR = 1
np.random.seed(42)
INTERVAL = 100
ITERATIONS = 3
SIM: Optional[Sim] = None
yaw_acc=None
yaw_rec=None
yaw_list=None
def kernel(a, b, l=1.0):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist / l**2)

def offset_scalar(x):
    return np.cos(x)/2

def x_discrete(x):
    yaw_acc = SIM.yaw_acc_mapping
    yaw_rec = SIM.yaw_vec_mapping
    yaw_list = SIM.yaw_list
    return min(yaw_list, key=lambda y: abs(x - y))
def f_discrete(x):
    return f_discrete_real_data(x)
def f_discrete_real_data(yaw):
    global SIM
    if SIM is None:
        SIM = None  # this line seems redundant as SIM is already None
    return SIM.sample("yaw_"+str(yaw)) if SIM else None
def offset_scalar_real_data(x):
    return np.cos(x)/2

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

def main_sim_data():
    optimizer = opt(n_iterations=ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                 kernel_scale=5, protection_width=1)
    print("regret")
    print(optimizer.get_cumulative_regret())
    x_train = optimizer.x_discrete(np.random.uniform(0, INTERVAL, 1))
    y_train = optimizer.f_discrete(x_train)
    x_test = np.linspace(0, INTERVAL, 100)

    x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test)

    benchmark(INTERVAL,ITERATIONS)

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, optimizer.f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill_between(x_test, mu_star - 1.9600 * var_star, mu_star + 1.9600 * var_star, color='b', alpha=.5,
                     label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("Offset")
    plt.show()
def main_real_data():
    global SIM
    SIM = Sim()

    optimizer = opt(n_iterations=ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                 kernel_scale=5, protection_width=1)
    print("regret")
    print(optimizer.get_cumulative_regret())
    x_train = optimizer.x_discrete(np.random.uniform(0, INTERVAL, 1))
    y_train = optimizer.f_discrete(x_train)
    x_test = np.linspace(0, INTERVAL, 100)

    x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test)

    #benchmark(INTERVAL,ITERATIONS)

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, optimizer.f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_test, mu_star, 'b-', label='Prediction')
    plt.fill_between(x_test, mu_star - 1.9600 * var_star, mu_star + 1.9600 * var_star, color='b', alpha=.5,
                     label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title("Offset")
    plt.show()

if __name__ == '__main__':
    main_real_data()

