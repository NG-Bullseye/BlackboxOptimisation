import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import time
from scipy.optimize import OptimizeResult
import random

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim

class RealDataOptimization:
    def __init__(self, app,maxiter, n_repeats):
        print("\n#################################################################################################")
        print("STARTING Custom_BO")
        self.n_repeats = n_repeats
        self.cumulative_regret_list = []
        self.n_points = 20  # number of benchmarks for avereging
        self.app = app
        self.obj_func = self.app.sampler.shift_dict_to_positive(
            self.app.sampler.yaw_acc)  # objective function solution. For Benchmarking purposes. Hidden to the optimization functions.
        self.bounds = [0, 90]
        self.start_time = None
        self.end_time = None
        np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
        sample_obj = app.sampler
        sample_obj.reset()  # Reset counter

        # Parameters for start_sim_with_real_data
        self.params = {
            'quantization_factor': 1,
            'kernel_scale': 0.09805098972579881,  # needs propper HP OPT and Training
            'offset_scale': 0.18080060285014135,  # needs propper HP OPT and Training
            'offset_range': 24.979610423583395,  # needs propper HP OPT and Training
            'protection_width': 0.8845792950045508,  # needs propper HP OPT and Training
            'n_iterations': maxiter,
            'randomseed': 524
        }

    def timeit(self, method):
        self.start_time = time.time()
        result = method()
        self.end_time = time.time()
        return result
#
    def cumulative_regret(self, all_evals):
        optimal_value = np.max(list(self.obj_func.values()))
        return np.sum(optimal_value - all_evals)
    def print_and_return_avg_metrics(self, optimal_xs, optimal_fxs, times, cum_regrets, n_evals):
       avg_cum_regret = np.mean(cum_regrets)
       avg_optimal_x = np.mean(optimal_xs)
       avg_optimal_fx = np.mean(optimal_fxs)
       avg_time = np.mean(times)
       global_optima_key = max(self.obj_func, key=self.obj_func.get)
       global_optima_value = self.obj_func[global_optima_key]
       print(f"Average Cumulative Regret over {self.n_points} runs: {avg_cum_regret}")
       print(f'Global optima: x={global_optima_key} y={global_optima_value}')
       return avg_optimal_x, avg_optimal_fx, avg_time, avg_cum_regret, n_evals


    def get_metrics(self, result_tuple):
        cum_regret = result_tuple
        optimal_x = self.app.optimal_x
        optimal_fx = self.app.optimal_fx
        total_time = self.end_time - self.start_time
        return optimal_x, optimal_fx, total_time, cum_regret

    def for_range_of_iter(self, maxiter):
        average_optimal_fxs=[]
        average_cum_regrets=[]

        for maxiter in range(0, maxiter + 1, 1):
            print(f"Running BO with maxiter = {maxiter}")

            cum_regrets = []
            optimal_xs = []
            optimal_fxs = []
            times = []
            for _ in range(self.n_repeats):
                current_time = time.perf_counter()
                randomseed = random.seed(int(current_time * 1e9))
                self.params['randomseed'] = randomseed
                results = self.timeit(lambda: self.app.start_sim_with_real_data(**self.params))
                optimal_x, optimal_fx, time_taken, cumulative_regret = self.get_metrics(results)
                cum_regrets.append(cumulative_regret)
                optimal_xs.append(optimal_x)
                optimal_fxs.append(optimal_fx)
                times.append(time_taken)

            avg_optimal_x, avg_optimal_fx, avg_time, avg_cum_regrets, n_evals =  self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times,cum_regrets,self.params["n_iterations"])
            average_optimal_fxs.append(avg_optimal_fx)
            average_cum_regrets.append(avg_cum_regrets)
            print(f"INFO: Appending {avg_optimal_fx} to average_optimal_fxs")
            print(f"CURRENT: maxiter: {maxiter}")
            print(f"CURRENT: average_optimal_fxs (index is iteration): {average_optimal_fxs}")
        self.plot_performance(maxiter,average_optimal_fxs)
        return average_optimal_fxs, average_cum_regrets

    def plot_performance(self,maxiter,average_optimal_fxs):
        global_max = self.app.sampler.getGlobalOptimum_Y()
        percentage_close_list = [(fx / global_max) * 100 for fx in average_optimal_fxs]

        #print(len(maxiter_list), len(percentage_close_list))

        plt.plot([0,maxiter], percentage_close_list, marker='o')
        if maxiter == 0:
            maxiter = 1
        # Set axis limits
        plt.xlim(0, maxiter)
        plt.ylim(0, 100)  # percentage can be up to 100

        # Ensure x-axis ticks are integers
        plt.xticks(np.arange(0, maxiter + 1, step=1))

        plt.title(f"BO")
        plt.xlabel("Max Iterations")
        plt.ylabel("Percentage Close to Global Max")
        plt.show()

    # Run the benchmark
def main(app,maxiter,n_repeats):
    # Run the benchmark
    benchmark = RealDataOptimization(app, n_repeats=n_repeats,maxiter=maxiter)
    avg_optimal_fxs,avg_cum_regrets = benchmark.for_range_of_iter(maxiter)
    print(f"Found optima: fx={avg_optimal_fxs}")
    print(f"With {n_repeats} repeats for every of the {maxiter} iterations ")
    return avg_optimal_fxs, avg_cum_regrets

if __name__ == '__main__':
    app=Application(Sampler(Sim()))
    avg_optimal_fxs,avg_cum_regrets= main(app,maxiter=1, n_repeats=10)
    print(f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {avg_optimal_fxs} \navg_cum_regrets:{avg_cum_regrets}")

