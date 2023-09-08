import numpy as np
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
        self.app = Application(Sampler( Sim()))
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

    #def get_metrics(self, result):
    #    total_time = self.end_time - self.start_time
    #    n_eval = self.app.sampler.function_call_counter
    #    self.app.sampler.function_call_counter = 0  # reset counter
#
    #    if isinstance(result, OptimizeResult):
    #        return result.x, -result.fun, n_eval, total_time
    #    elif isinstance(result, tuple):
    #        # handle the tuple, maybe it contains result and evaluation
    #        result_obj, evaluation = result
    #        return result_obj.x, -result_obj.fun, n_eval, total_time
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
       return avg_optimal_x, avg_optimal_fx, avg_time, cum_regrets, n_evals
#    def print_and_return_avg_metrics(self, optimal_xs, optimal_fxs, times, cum_regrets,n_evals):
#        avg_cum_regret = np.mean(cum_regrets)
#        avg_optimal_x = np.mean(optimal_xs)
#        avg_optimal_fx = np.mean(optimal_fxs)
#        avg_time = np.mean(times)
#        avg_evals = np.mean(n_evals)
#        global_optima_key = max(self.obj_func, key=self.obj_func.get)
#        global_optima_value = self.obj_func[global_optima_key]
#        print(f"Average Cumulative Regret over {self.n_points} runs: {avg_cum_regret}")
#        print(f'Global optima: x={global_optima_key} y={global_optima_valu


    def get_metrics(self, result_tuple):
        cum_regret = result_tuple
        optimal_x = self.app.optimal_x
        optimal_fx = self.app.optimal_fx
        total_time = self.end_time - self.start_time
        return optimal_x, optimal_fx, total_time, cum_regret


    def run(self):
        cum_regrets = []
        optimal_xs = []
        optimal_fxs = []
        times = []

        for _ in range(self.n_repeats):
            current_time = time.perf_counter()
            randomseed = random.seed(int(current_time * 1e9))
            self.params['randomseed']=randomseed
            results = self.timeit(lambda: self.app.start_sim_with_real_data(**self.params))
            optimal_x, optimal_fx, time_taken, cumulative_regret = self.get_metrics(results)
            cum_regrets.append(cumulative_regret)
            optimal_xs.append(optimal_x)
            optimal_fxs.append(optimal_fx)
            times.append(time_taken)

        return self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times,cum_regrets,self.params["n_iterations"])





    # Run the benchmark
def main(maxiter,n_repeats):
    # Run the benchmark
    benchmark = RealDataOptimization(Application(Sampler(Sim())), n_repeats=n_repeats,maxiter=maxiter)
    avg_optimal_x, avg_optimal_fx, avg_time, cum_regrets, n_evals = benchmark.run()
    print(f"Found optima: x={avg_optimal_x} y={avg_optimal_fx}")
    print(f"With {n_evals} Evaluations")
    print(f"Average time taken: {avg_time}")
    return avg_optimal_fx, cum_regrets
if __name__ == '__main__':
    print(f"FINAL RESULTS CUSTOM BO: {main(maxiter= 10,n_repeats=100)}")

