import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import time
from scipy.optimize import OptimizeResult

class Benchmark(ABC):
    def __init__(self, func, bounds):
        self.func = func
        self.bounds = bounds
        self.func.counter = 0
        self.start_time = None
        self.end_time = None

    def timeit(self, method):
        self.start_time = time.time()
        result = method()
        self.end_time = time.time()
        return result

    def get_metrics(self, result):
        total_time = self.end_time - self.start_time
        n_eval = self.func.counter
        self.func.counter = 0  # reset counter
        return result.x, -result.fun, n_eval, total_time

    def cumulative_regret(self, all_evals):
        optimal_value = np.max(all_evals)
        return np.sum(optimal_value - all_evals)

    @abstractmethod
    def run(self):
        pass
class Optimization(Benchmark):
    def __init__(self, func, bounds, x0, iterations):
        super().__init__(func, bounds)
        self.x0 = x0
        self.iterations = iterations

    def run(self):
        def method():
            result = minimize(self.func, self.x0, bounds=self.bounds, method='L-BFGS-B',
                              options={'maxiter': self.iterations})
            return result, self.func(result.x)  # return evaluation along with result

        result, evaluation = self.timeit(method)
        print(f"Cumulative regret: {self.cumulative_regret([evaluation])}")
        return self.get_metrics(result)

class GridSearch(Benchmark):
    def __init__(self, func, bounds, n_points):
        super().__init__(func, bounds)
        self.grid = np.linspace(*bounds[0], n_points)

    def run(self):
        def method():
            grid_eval = np.array([self.func(x) for x in self.grid])
            max_index = np.argmax(-grid_eval)
            return OptimizeResult(x=self.grid[max_index],
                                  fun=-grid_eval[max_index]), grid_eval  # return all evaluations along with result

        result, all_evals = self.timeit(method)
        print(f"Cumulative regret: {self.cumulative_regret(all_evals)}")
        return self.get_metrics(result)

class RandomSearch(Benchmark):
    def __init__(self, func, bounds, n_points):
        super().__init__(func, bounds)
        self.n_points = n_points

    def run(self):
        def method():
            random_points = np.random.uniform(*self.bounds[0], self.n_points)
            random_eval = np.array([self.func(x) for x in random_points])
            max_index = np.argmax(-random_eval)
            return OptimizeResult(x=random_points[max_index],
                                  fun=-random_eval[max_index]), random_eval  # return all evaluations along with result

        result, all_evals = self.timeit(method)
        print(f"Cumulative regret: {self.cumulative_regret(all_evals)}")
        return self.get_metrics(result)

