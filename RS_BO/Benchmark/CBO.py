import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import time
from scipy.optimize import OptimizeResult
import random

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim


class RealDataOptimization:
    def __init__(self, app, maxiter, n_repeats):
        self.app = app
        self.n_repeats = n_repeats
        self.start_time, self.end_time = None, None
        self.params = {
            'quantization_factor': 1,
            'kernel_scale': 0.09805098972579881,
            'offset_scale': 0.18080060285014135,
            'offset_range': 24.979610423583395,
            'protection_width': 0.8845792950045508,
            'n_iterations': maxiter,
            'randomseed': 524
        }

    def timeit(self, method):
        self.start_time = time.time()
        result = method()
        self.end_time = time.time()
        return result

    def get_metrics(self, result_tuple):
        cum_regret = result_tuple
        optimal_x = self.app.optimal_x
        optimal_fx = self.app.optimal_fx
        total_time = self.end_time - self.start_time
        return optimal_x, optimal_fx, total_time, cum_regret

    def run_CBO_multiple_iterations(self):
        # Placeholder: this method will run CBO multiple times
        pass

    def CBO(self):
        # Placeholder: this method will implement your Custom Bayesian Optimization
        pass

    def plot_performance(self):
        # Placeholder: this method will plot the performance
        pass


def main(maxiter, n_repeats):
    app = Application(Sampler(Sim()))
    benchmark = RealDataOptimization(app, maxiter=maxiter, n_repeats=n_repeats)
    # Call run_CBO_multiple_iterations or any other methods you'd like
    return "Finished running."


if __name__ == '__main__':
    print(f"FINAL RESULTS CUSTOM BO: {main(maxiter=10, n_repeats=100)}")
