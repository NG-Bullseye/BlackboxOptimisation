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
        self.INTERVAL = 45
        self.shift_value = -45 #set to the minimum of the negative value interval of sampled data
        self.ITERATIONS = 10
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
        original_yaws=self.shift_to_original(shifted_yaws)
        for yaw in original_yaws:
            yaw_value = self.x_discrete_real_data_unshifted([yaw])# Get the single value from the returned list
            yaw_value = yaw_value[0]
            sampled_acc = self.SIM.sample(str(yaw_value))
            sampled_accs.append(sampled_acc)
        return sampled_accs

    def x_discrete_real_data_unshifted(self, yaws):
        if not isinstance(yaws, (np.ndarray, list, tuple)):
            yaws = np.array([yaws])
        result = []
        for yaw_value in yaws:
            closest_yaw = None
            min_diff = float('inf')  # Initialize with infinity
            positive_yaw_list = self.yaw_list
            for yaw in positive_yaw_list:
                diff = abs(yaw_value - yaw)
                if diff < min_diff:
                    min_diff = diff
                    closest_yaw = yaw

            result.append(closest_yaw)
        return result
    def x_discrete_real_data(self, yaws):
        if not isinstance(yaws, (np.ndarray, list, tuple)):
            yaws = np.array([yaws])
        result = []
        for yaw_value in yaws:
            closest_yaw = None
            min_diff = float('inf')  # Initialize with infinity
            positive_yaw_list=self.shift_to_positive(self.yaw_list)
            for yaw in positive_yaw_list:
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

    def shift_to_positive(self, x_values):
        x_values = np.array(x_values)  # Convert input to numpy array


        if self.shift_value < 0:  # Just check the scalar value directly
            x_values -= self.shift_value  # Shift all values to positive

        return x_values

    def shift_to_original(self, x_values):
        if self.shift_value is None:
            raise ValueError("shift_value has not been set. Please run shift_to_positive first.")

        x_values = np.array(x_values)  # Convert to numpy array for consistency
        if self.shift_value < 0:
            x_values += self.shift_value  # Reverse the shift to get original values

        return x_values

    def start_sim_with_test_data(self):
        optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                     kernel_scale=3, protection_width=1)

        x_train = self.x_discrete(np.random.uniform(0, self.INTERVAL, 1))

        y_train = self.f_discrete(x_train)

        x_test = np.linspace(0, self.INTERVAL, 100)

        x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test,self.f_discrete ,self.x_discrete)

        #self.benchmark(self.INTERVAL,self.ITERATIONS)

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
        optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range= 1, offset_scale=1,
                     kernel_scale=2, protection_width=1)
        x_train = self.x_discrete_real_data_unshifted(np.random.uniform(-45, self.INTERVAL, 1)) #shifts all value into the positive x direction so that the algorithm. Required for the optimization algo
        x_train_pos = self.shift_to_positive(x_train)
        y_train_pos = np.array( self.f_discrete_real_data(x_train_pos))
        x_test = np.sort(np.array(self.SIM.get_all_yaw_values())) #here the sim db yaw as list must be insearted maybe?
        x_test = self.shift_to_positive(x_test)
        x_train_pos, y_train_pos, mu_star, var_star = optimizer.optimize(x_train_pos, y_train_pos, x_test, self.f_discrete_real_data, self.x_discrete_real_data)
        #self.benchmark(self,INTERVAL,ITERATIONS)

        #sorted_train_data = sorted(zip(x_train, y_train))
        #x_train_sorted, y_train_sorted = zip(*sorted_train_data)
        #sorted_test_data = sorted(zip(x_test, mu_star,var_star))
        #print(sorted_test_data)
        #x_test_sorted, mu_star_sorted, var_star_sorted = zip(*sorted_test_data)
        #print("mu_star_sorted:", mu_star_sorted)
        #mu_star_sorted =  np.array(list(mu_star_sorted))
        #var_star_sorted =  np.array(list(var_star_sorted))

        x_test_sorted=x_test
        x_train_sorted=x_train_pos
        y_train_sorted=y_train_pos
        mu_star_sorted=mu_star
        var_star_sorted=var_star

        plt.figure(figsize=(12, 8))
        y_test = np.array(self.f_discrete_real_data(x_test_sorted))
        x_ticks = np.linspace(-45, 45, 20)  # 20 ticks
        plt.xticks(x_ticks)
        x_test_sorted_og= self.shift_to_original(x_test_sorted)
        x_train_sorted_og=self.shift_to_original(x_train_sorted)
        plt.plot(x_test_sorted_og, y_test , 'r:', label='Sampled Data')

        plt.plot(x_train_sorted_og, y_train_sorted, 'r.', markersize=10, label='Observations')

        plt.plot(x_test_sorted_og, mu_star_sorted, 'b-', label='Prediction')

        plt.fill_between(x_test_sorted_og, mu_star_sorted- 1.9600 * var_star_sorted , mu_star_sorted + 1.9600 * var_star_sorted, color='b', alpha=.5, label='95% confidence interval')
        plt.xlabel('$yaw$')
        plt.ylabel('$acc$')
        plt.legend(loc='upper left')
        plt.title("Offset")
        plt.show()

if __name__ == '__main__':
    app = Application()
    import os
    b=os.environ.get("test")
    print("Environment Variable:", b)  # Debug print
    if b=='1':
        app.start_sim_with_test_data()
    else:
        app.start_sim_with_real_data()

