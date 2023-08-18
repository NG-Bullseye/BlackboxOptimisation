import numpy as np
from RS_BO.Custom_Gaussian import Optimization as opt
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional

from RS_BO.Benchmark import Benchmark, Optimization, GridSearch, RandomSearch
from RS_BO.Utility.Sim import Sim

class Application():
    def __init__(self):
        self.shift_value = 0
        np.random.seed(42)
        self.QUANTIZATION_FACTOR = 1
        self.INTERVAL = 100
        self.ITERATIONS = 3
        self.SIM = Sim()
        self.yaw_acc = self.SIM.yaw_acc_mapping
        self.yaw_rec = self.SIM.yaw_vec_mapping
        self.yaw_list = self.SIM.yaw_list
        self.f_discrete_counter_counter = 0
    def reset_f_discrete_counter_counter(self):
        self.f_discrete_counter_counter = 0
    def kernel(self,a, b, l=1.0):
        sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist / l**2)

    def offset_scalar(x):
        return np.cos(x) / 2
    def x_discrete(self,x):
        return np.round(x / self.QUANTIZATION_FACTOR) * self.QUANTIZATION_FACTOR
    def f_discrete(self,x):
        sampled_acc = (np.sin(self.x_discrete(x)) + 1) / 2
        return sampled_acc

    def f_discrete_real_data(self, shifted_yaws):
        sampled_accs = []
        original_yaws=shifted_yaws#self.shift_to_original(shifted_yaws)
        for yaw in original_yaws:
            yaw_value = self.x_discrete_real_data([yaw])# Get the single value from the returned list
            yaw_value = yaw_value[0]
            sampled_acc = self.SIM.sample(str(yaw_value))
            sampled_accs.append(sampled_acc)
        return sampled_accs

    def x_discrete_real_data(self, yaws):
        if not isinstance(yaws, (np.ndarray, list, tuple)):
            yaws = np.array([yaws])
        result = []
        for yaw_value in yaws:
            closest_yaw = None
            min_diff = float('inf')  # Initialize with infinity

            for yaw in self.yaw_list:
                diff = abs(yaw_value - yaw)
                if diff < min_diff:
                    min_diff = diff
                    closest_yaw = yaw

            result.append(closest_yaw)
        return result

    def offset_scalar_real_data(self,x):
        return np.cos(x)/2

    # Add function call counter to f_discrete
    def f_discrete_counter(self,x):
        self.f_discrete_counter_counter += 1
        return self.f_discrete(x)

    def plot_gp(self,gp, x_test, f_discrete):
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
    def benchmark(self,INTERVAL,ITERATIONS):
        benchmark_classes = [Optimization, GridSearch, RandomSearch]
        for BenchmarkClass in benchmark_classes:
            if BenchmarkClass == Optimization:
                benchmark = BenchmarkClass(self.f_discrete_counter, [(0, INTERVAL)], 0,ITERATIONS)
            else:
                benchmark = BenchmarkClass(self.f_discrete_counter, [(0, INTERVAL)], ITERATIONS)

            optimal_x, optimal_fx, n_eval, time_taken = benchmark.run()

            print(f"\n{BenchmarkClass.__name__} Results:")
            print(f"The optimal value of x is {optimal_x}")
            print(f"The optimal value of f(x) is {optimal_fx}")
            print(f"Time taken: {time_taken}")

    def shift_to_positive(self,x_values):
        self.shift_value = min(0, min(x_values))
        if self.shift_value < 0:
            x_values = x_values - self.shift_value
        return x_values, self.shift_value

    def shift_to_original(self,x_values):
        if self.shift_value < 0:
            x_values = [x + self.shift_value for x in x_values]
        return x_values

    def start_sim_with_test_data(self):
        optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                     kernel_scale=5, protection_width=1)
        x_train = self.x_discrete(np.random.uniform(0, self.INTERVAL, 1))
        y_train = self.f_discrete(x_train)
        x_test = np.linspace(0, self.INTERVAL, 100)

        x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test,self.f_discrete ,self.x_discrete)

        self.benchmark(self.INTERVAL,self.ITERATIONS)

        plt.figure(figsize=(12, 8))
        plt.plot(x_test, self.f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
        plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
        plt.plot(x_test, mu_star, 'b-', label='Prediction')
        plt.fill_between(x_test, mu_star - 1.9600 * var_star, mu_star + 1.9600 * var_star, color='b', alpha=.5,
                         label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend(loc='upper left')
        plt.title("Offset")
        plt.show()
    def start_sim_with_real_data(self):
        optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range= 5, offset_scale=0.1,
                     kernel_scale=5, protection_width=1)
        x_train = self.x_discrete_real_data(np.random.uniform(0, self.INTERVAL, 1))
        #x_train = self.shift_to_positive(x_train) #shifts all value into the positive x direction so that the algorithm. Required for the optimization algo
        y_train = np.array( self.f_discrete_real_data(x_train))
        x_test = np.array(self.SIM.get_all_yaw_values()) #here the sim db yaw as list must be insearted maybe?

        x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test, self.f_discrete_real_data, self.x_discrete_real_data)
        #self.benchmark(self,INTERVAL,ITERATIONS)
        sorted_train_data = sorted(zip(x_train, y_train))
        x_train_sorted, y_train_sorted = zip(*sorted_train_data)
        sorted_test_data = sorted(zip(x_test, mu_star))
        x_test_sorted, mu_star_sorted = zip(*sorted_test_data)

        plt.figure(figsize=(12, 8))
        y_test = self.f_discrete_real_data(x_test_sorted)
        plt.plot(x_test_sorted, y_test, 'r:', label='Sampled Data')

        #plt.plot(x_train_sorted, y_train_sorted, 'r.', markersize=10, label='Observations')

        #plt.plot(x_test_sorted, mu_star_sorted, 'b-', label='Prediction')

        #plt.fill_between(x_test_sorted, mu_star_sorted - 1.9600 * var_star, mu_star_sorted + 1.9600 * var_star, color='b', alpha=.5, label='95% confidence interval')
        plt.xlabel('$yaw$')
        plt.ylabel('$acc$')
        plt.legend(loc='upper left')
        plt.title("Offset")
        plt.show()

if __name__ == '__main__':
    app = Application()
    #app.start_sim_with_test_data()
    app.start_sim_with_real_data()
