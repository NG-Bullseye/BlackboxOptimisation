from tqdm import tqdm  # import tqdm
from skopt import gp_minimize
from skopt.space import Real
import numpy as np
from RS_BO.Utility.Sim import Sim
from Application import Application, Sampler
import random
import time

min_cumulative_regret_global = float('inf')
data = Sampler(Sim())
class MyApp:
    def start_optimization(self, quantization_factor, offset_range, offset_scale, kernel_scale, protection_width,
                           n_iterations):
        return np.random.rand() * 10

def objective(params):
    kernel_scale, offset_scale, offset_range, protection_width = params
    n_iterations_start = 10
    montecarlo_iter = 3
    average_cumulative_regret_sum = 0.
    n_iterations = n_iterations_start

    for i in range(montecarlo_iter):
        current_time = time.perf_counter()
        randomseed=random.seed(int(current_time * 1e9))
        app = Application(data)
        cumulative_regret = app.start_sim_with_real_data(1, offset_range=offset_range, offset_scale=offset_scale, kernel_scale=kernel_scale, protection_width=protection_width,n_iterations=n_iterations,randomseed=randomseed)
        average_cumulative_regret_sum += cumulative_regret

    average_cumulative_regret = average_cumulative_regret_sum / montecarlo_iter
    return average_cumulative_regret

space = [
    Real(0.005, 1.0, name="kernel_scale"),
    Real(0.01, 1.0, name="offset_scale"),
    Real(20, 50.0, name="offset_range"),
    Real(0.05, 2.0, name="protection_width")
]

n_calls = 1000
x0 = [[0.09805098972579881, 0.18080060285014135, 24.979610423583395, 0.8845792950045508]]

# Wrap the gp_minimize call with tqdm to add a progress bar
with tqdm(total=n_calls, desc="Optimizing", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    def callback(res):
        pbar.update(1)

    res = gp_minimize(objective, space, n_calls=n_calls, acq_func="EI", x0=x0, n_jobs=-1, callback=callback)

best_params = res.x
print(f"Best parameters: {best_params} min_cumulative_regret_global: {min_cumulative_regret_global}")
