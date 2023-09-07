import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import time
from scipy.optimize import OptimizeResult
import random

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim


class Benchmark(ABC):
    def __init__(self, app, bounds):
        self.n_points = 20 #number of benchmarks for avereging
        self.app = app
        self.obj_func=self.app.sampler.shift_dict_to_positive(self.app.sampler.yaw_acc)  # objective function solution. For Benchmarking purposes. Hidden to the optimization functions.
        self.bounds = bounds
        self.start_time = None
        self.end_time = None

    def timeit(self, method):
        self.start_time = time.time()
        result = method()
        self.end_time = time.time()
        return result

    def get_metrics(self, result):
        total_time = self.end_time - self.start_time
        n_eval = self.app.sampler.function_call_counter
        self.app.sampler.function_call_counter = 0  # reset counter

        if isinstance(result, OptimizeResult):
            return result.x, -result.fun, n_eval, total_time
        elif isinstance(result, tuple):
            # handle the tuple, maybe it contains result and evaluation
            result_obj, evaluation = result
            return result_obj.x, -result_obj.fun, n_eval, total_time

    def cumulative_regret(self, all_evals):
        optimal_value = np.max(list(self.obj_func.values()))
        return np.sum(optimal_value - all_evals)
    def print_and_return_avg_metrics(self, optimal_xs, optimal_fxs, times, cum_regrets,n_evals):
        avg_cum_regret = np.mean(cum_regrets)
        avg_optimal_x = np.mean(optimal_xs)
        avg_optimal_fx = np.mean(optimal_fxs)
        avg_time = np.mean(times)
        avg_evals = np.mean(n_evals)
        global_optima_key = max(self.obj_func, key=self.obj_func.get)
        global_optima_value = self.obj_func[global_optima_key]
        print(f"Average Cumulative Regret over {self.n_points} runs: {avg_cum_regret}")
        print(f'Global optima: x={global_optima_key} y={global_optima_value}')
        return avg_optimal_x, avg_optimal_fx, avg_time,avg_cum_regret,avg_evals

    @abstractmethod
    def run(self):
        pass

class RealDataOptimization(Benchmark):
    def __init__(self, app, n_repeats, params):
        print("\n#################################################################################################")
        print("STARTING Custom_BO")
        self.app = app
        self.n_repeats = n_repeats
        self.params = params
        self.cumulative_regret_list = []
        super().__init__(app, None)

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

        return self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times,cum_regrets,params["n_iterations"])
    def get_metrics(self, result_tuple):
        cum_regret = result_tuple
        optimal_x = application.optimal_x
        optimal_fx = application.optimal_fx
        total_time = self.end_time - self.start_time
        return optimal_x, optimal_fx, total_time, cum_regret

    def print_and_return_avg_metrics(self, optimal_xs, optimal_fxs, times, cum_regrets,n_evals):
        avg_cum_regret = np.mean(cum_regrets)
        avg_optimal_x = np.mean(optimal_xs)
        avg_optimal_fx = np.mean(optimal_fxs)
        avg_time = np.mean(times)
        global_optima_key = max(self.obj_func, key=self.obj_func.get)
        global_optima_value = self.obj_func[global_optima_key]
        print(f"Average Cumulative Regret over {self.n_points} runs: {avg_cum_regret}")
        print(f'Global optima: x={global_optima_key} y={global_optima_value}')
        return avg_optimal_x, avg_optimal_fx, avg_time,cum_regrets,n_evals


class GridSearch(Benchmark):
    def __init__(self, func, bounds, n_repeats,n_points):
        super().__init__(func, bounds)
        print("\n#################################################################################################")
        print("STARTING GridSearch")
        self.grid = np.linspace(*bounds[0], n_points)
        self.n_points = n_points  # Adding this line to make it consistent with RandomSearch
        self.n_repeats = n_repeats

    def run(self):
        cum_regrets = []
        optimal_xs = []
        optimal_fxs = []
        n_evals = []
        times = []

        for _ in range(self.n_points):
            result, all_evals = self.timeit(self.method)
            optimal_x, optimal_fx, n_eval, time_taken = self.get_metrics(result)

            cum_regrets.append(self.cumulative_regret(all_evals))
            optimal_xs.append(optimal_x)
            optimal_fxs.append(optimal_fx)
            n_evals.append(n_eval)
            times.append(time_taken)

        return self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times, cum_regrets, n_evals)

    def method(self):
        grid_eval = np.array([self.app.sampler.f_discrete_real_data(x) for x in self.grid])
        max_index = np.argmax(-grid_eval)
        return OptimizeResult(x=self.grid[max_index], fun=-grid_eval[max_index]), grid_eval

class RandomSearch(Benchmark):
    def __init__(self, func, bounds, n_repeats, n_points):
        super().__init__(func, bounds)
        self.n_points = n_points
        self.n_repeats = n_repeats
        print("\n#################################################################################################")
        print("STARTING RandomSearch")

    def run(self):
        cum_regrets = []
        optimal_xs = []
        optimal_fxs = []
        n_evals = []
        times = []

        for _ in range(self.n_points):
            result, all_evals = self.timeit(self.method)
            optimal_x, optimal_fx, n_eval, time_taken = self.get_metrics(result)

            cum_regrets.append(self.cumulative_regret(all_evals))
            optimal_xs.append(optimal_x)
            optimal_fxs.append(optimal_fx)
            n_evals.append(n_eval)
            times.append(time_taken)

        return self.print_and_return_avg_metrics(optimal_xs, optimal_fxs, times, cum_regrets, n_evals)

    def method(self):
        random_points = np.random.uniform(*self.bounds[0], self.n_points)
        random_eval = np.array([self.app.sampler.f_discrete_real_data(x) for x in random_points])
        max_index = np.argmax(-random_eval)
        return OptimizeResult(x=random_points[max_index], fun=-random_eval[max_index]), random_eval

if __name__ == '__main__':
    np.random.seed(random.seed(int( time.perf_counter() * 1e9)))
    application = Application(Sampler(0.1, Sim()))
    sample_obj = application.sampler
    sample_obj.reset()  # Reset counter
    bounds = [(0, 90)]
    ITERATIONS = 10
    # Parameters for start_sim_with_real_data
    params = {
        'quantization_factor': 1,
        'kernel_scale': 0.09805098972579881,
        'offset_scale': 0.18080060285014135,
        'offset_range': 24.979610423583395,
        'protection_width': 0.8845792950045508,
        'n_iterations': 10,
        'randomseed': 524
    }
    n_repeats=10
    # Run the benchmark
    benchmark_classes = [RealDataOptimization, GridSearch, RandomSearch]
    for BenchmarkClass in benchmark_classes:
        if BenchmarkClass == RealDataOptimization:
            benchmark = BenchmarkClass(application,n_repeats=n_repeats,params=params)
        else:
            benchmark = BenchmarkClass(application, bounds, n_repeats=n_repeats,n_points=params['n_iterations'])
        avg_optimal_x, avg_optimal_fx, avg_time, cum_regrets,n_evals = benchmark.run()
        print(f"Found optima: x={avg_optimal_x} y={avg_optimal_fx}")
        print(f"With {n_evals} Evaluations")
        print(f"Average time taken: {avg_time}")
