from datetime import datetime

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
    def __init__(self, app,maxiter, n_repeats,deactivate_rec_scalar=False):
        print("\n#################################################################################################")
        print("STARTING Custom_BO")
        self.deactivate_rec_scalar=deactivate_rec_scalar
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
        self.maxiter=maxiter

        # Parameters for start_sim_with_real_data
        self.params = {
            'quantization_factor': 1,
            'kernel_scale': 6.021461291655982,  # 0.09805098972579881,  # 2.221461291655982
            'offset_scale': 0.0253434383380036,  # 0.18080060285014135,  # 0.6953434383380036
            'offset_range': 30.92663911701745,  # 24.979610423583395,  #  50.92663911701745
            'protection_width': 3.2715701918611297,  # 0.8845792950045508,  #  3.2715701918611297
            'n_iterations': 0,
            'randomseed': 524,
            'deactivate_rec_scalar': False
        }

    def timeit(self, method):
        self.start_time = time.time()
        result = method()
        self.end_time = time.time()
        return result
#
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
    def track_variance(self,max_found_y_list):
        return np.var(max_found_y_list)
    def for_range_of_iter(self,plotting=False):
        average_optimal_fxs=[]
        average_cum_regrets=[]
        var_fxs = []
        for iter in range(0,self.maxiter):
            times = []
            avg_optimal_fx = 0
            avg_cum_reg = 0
            max_found_y_values = []
            for _ in range(self.n_repeats):
                self.app.sampler.evaluated_x=[]
                current_time = time.perf_counter()
                randomseed = random.seed(int(current_time * 1e9))
                self.params['randomseed'] = randomseed
                self.params['n_iterations']=iter
                self.params['plotting']=plotting
                results = self.timeit(lambda: self.app.start_sim_with_real_data(**self.params))
                optimal_x, max_found_y, time_taken, cumulative_regret = self.get_metrics(results)
                max_found_y_values.append(max_found_y)
                times.append(time_taken)
                avg_optimal_fx += max_found_y  # Update it directly here
                avg_cum_reg += cumulative_regret
            var_fxs.append(self.track_variance(max_found_y_values))
            avg_optimal_fx /= self.n_repeats
            avg_cum_reg /= self.n_repeats
            average_optimal_fxs.append(avg_optimal_fx)  # Directly append to list
            average_cum_regrets.append(avg_cum_reg)

            #avg_optimal_x, avg_optimal_fx, avg_time, avg_cum_regrets, n_evals =  self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times,cum_regrets,self.params["n_iterations"])
            #average_optimal_fxs.append(avg_optimal_fx)
            #average_cum_regrets.append(avg_cum_regrets)
            print(f"INFO: Appending {avg_optimal_fx} to average_optimal_fxs")
            print(f"CURRENT: maxiter: {self.maxiter}")
            print(f"CURRENT: average_optimal_fxs (index is iteration): {average_optimal_fxs}")
        #self.plot_performance(maxiter,average_optimal_fxs)
        return average_optimal_fxs, average_cum_regrets,var_fxs


    def plot_graph(self, iterations,CBO_fxs):
        scale=1
        fig, ax1 = plt.subplots(figsize=(10 * scale, 6 * scale))

        color_list = ['b', 'g', 'r', 'm']
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel(f'Average Accuracy found over {self.n_repeats} repeats', color='k')
        plt.title(f"Global maximum {self.app.sampler.getGlobalOptimum_Y()} ")
        print(CBO_fxs, iterations)
        ax1.plot(range(0,iterations+1), CBO_fxs, color=color_list[2], label='CBO_fxs')

        ax1.legend(loc='upper left')
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Graphs_{timestamp}"
        #self.save_plot(folder_name, 'optimal_fxs_plot.png')
        plt.show()
    def plot_performance(self,maxiter,average_optimal_fxs):
        global_max = self.app.sampler.getGlobalOptimum_Y()

        #print(len(maxiter_list), len(percentage_close_list))
        #iteration_list
        plt.plot([0,maxiter], average_optimal_fxs, marker='o')
        if maxiter == 0:
            maxiter = 1
        # Set axis limits
        plt.xlim(0, maxiter)
        plt.ylim(-1, 100)  # percentage can be up to 100

        # Ensure x-axis ticks are integers
        plt.xticks(np.arange(0, maxiter + 1, step=1))

        plt.title(f"BO")
        plt.xlabel("Max Iterations")
        plt.ylabel("Percentage Close to Global Max")
        plt.show()

    # Run the benchmark
def main(app,maxiter,n_repeats,plotting=False):
    # Run the benchmark
    cboMain = RealDataOptimization(app, n_repeats=n_repeats,maxiter=maxiter+1)
    avg_optimal_fxs,avg_cum_regrets ,var_fxs= cboMain.for_range_of_iter(plotting)
    print(f"Found optima: fx={avg_optimal_fxs}")
    print(f"With {n_repeats} repeats for every of the {maxiter} iterations ")
    if plotting:
        #benchmark.plot_performance(maxiter,avg_optimal_fxs)
        cboMain.plot_graph(maxiter, avg_optimal_fxs)
    return avg_optimal_fxs, avg_cum_regrets ,var_fxs
def main_no_rec(app,maxiter,n_repeats,plotting=False):
    # Run the benchmark
    cbo = RealDataOptimization(app, n_repeats=n_repeats, maxiter=maxiter + 1,deactivate_rec_scalar=True)
    avg_optimal_fxs, avg_cum_regrets,var_fxs = cbo.for_range_of_iter()
    print(f"Found optima: fx={avg_optimal_fxs}")
    print(f"With {n_repeats} repeats for every of the {maxiter} iterations ")
    if plotting:
        # benchmark.plot_performance(maxiter,avg_optimal_fxs)
        cbo.plot_graph(maxiter, avg_optimal_fxs)
    return avg_optimal_fxs, avg_cum_regrets,var_fxs

if __name__ == '__main__':
    app=Application(Sampler(Sim("Testdata")))
    maxiter = 4
    n_repeats = 0
    avg_optimal_fxs,avg_cum_regrets,var_fxs= main(app,maxiter, n_repeats,plotting=True)
    print(f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {avg_optimal_fxs} \navg_cum_regrets:{avg_cum_regrets}")

#FINAL RESULTS CUSTOM BO:
#avg_optimal_fxs: [0.8074000000000008, 0.8799500000000003]
#avg_cum_regrets:[0.17259999999999986, 0.34159999999999996]

#FINAL RESULTS CUSTOM BO no_rec:
#avg_optimal_fxs: [0.8196500000000003, 0.8736000000000008]
#avg_cum_regrets:[0.16034999999999988, 0.35314999999999996]