import numpy as np
from Custom_Gaussian import Optimization as opt
import matplotlib.pyplot as plt
from datetime import datetime

from RS_BO.Benchmark import Benchmark, Optimization, GridSearch, RandomSearch
class DataClass:

    def __init__(self):
        self.yaw_acc_mapping = load_real_data.main(INPUT_DATAPATH)
        self.optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                        kernel_scale=5, protection_width=1)
    def main_real_data(self, INPUT_DATAPATH):



        x_train = optimizer.x_discrete(np.random.uniform(0, self.INTERVAL, 1))
        y_train = optimizer.f_discrete(x_train)
        x_test = np.linspace(0, self.INTERVAL, 100)

        x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test)

        self.benchmark(self.INTERVAL, self.ITERATIONS)

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