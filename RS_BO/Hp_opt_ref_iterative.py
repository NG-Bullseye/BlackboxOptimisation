from tqdm import tqdm
from skopt import gp_minimize
from RS_BO.Utility.Sim import Sim
from RS_BO.Application import Application, Sampler
import random
import time
import numpy as np

class Optimizer:
    def __init__(self):
        self.data = Sampler(Sim())
        self.last_best_params = None
        self.original_bounds = [
            [0.005, 20.0], #kernel_scale
            [1, 10.0],#offset_scale
            [1, 40.0], #offset_range
            [1, 10.0] #protection_width
        ]

    def check_and_recenter(self, best_params, n_calls):
        new_bounds = self.original_bounds.copy()
        restart = False
        converged = False
        for i, param in enumerate(best_params):
            low, high = self.original_bounds[i]

            if abs(param - high) < 1:
                high = high * 2
            elif abs(param - low) < 1e-2:
                restart = True

            mid = param
            new_low = 10 ** (np.log10(mid) - 1)
            new_high = 10 ** (np.log10(mid) + 1)

            self.original_bounds[i] = [new_low, min(new_high, high)]

        if self.last_best_params:
            changes = [(abs(a - b) / ((a + b) / 2)) * 100 for a, b in zip(self.last_best_params, best_params)]
            if all(change < 5 for change in changes):
                print("Optimization converged. Less than 5% change.")
                converged = True

        if converged:
            return True, self.original_bounds

        if restart:
            print("Parameters near bounds, recentering and restarting...")
            return False, new_bounds
        else:
            print(f"Rerunning with {int((n_calls * 1.5))} calls.")
            return False, self.original_bounds

    def objective(self, params):
        offset_scale, offset_range, protection_width = params
        montecarlo_iter = 3
        average_cumulative_regret_sum = 0.
        for i in range(montecarlo_iter):
            current_time = time.perf_counter()
            randomseed = random.seed(int(current_time * 1e9))
            app = Application(self.data)
            cumulative_regret = app.start_sim_with_real_data(1,
                                                             kernel_scale=2.221461291655982,
                                                             offset_scale=offset_scale,
                                                             offset_range=offset_range,
                                                             protection_width=protection_width,
                                                             n_iterations=10,
                                                             randomseed=randomseed)
            average_cumulative_regret_sum += cumulative_regret
        average_cumulative_regret = average_cumulative_regret_sum / montecarlo_iter
        return average_cumulative_regret

    def run(self, n_calls, bounds=None, x0=None):
        converged = False
        while not converged:
            bounds = self.original_bounds.copy()
            with tqdm(total=n_calls, desc="Optimizing", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
                def callback(res):
                    pbar.update(1)

                # Make sure all bounds and x0 values are float
                bounds = [[float(low), float(high)] for low, high in bounds]
                if x0:
                    x0 = [float(x) for x in x0]

                # Print x0 and bounds for debugging
                print(f"Start Opt \nx0: {x0}, \nbounds: {bounds}")

                res = gp_minimize(self.objective, bounds, n_calls=n_calls, acq_func="EI", x0=x0, n_jobs=-1,
                                  callback=callback)
                best_params = res.x
                print(f"Best parameters: {best_params}")
                converged, new_bounds = self.check_and_recenter(best_params, n_calls)
                if converged:
                    break
                n_calls = int(n_calls * 1.5)
                bounds = new_bounds
                x0 = best_params



if __name__ == "__main__":
    opt = Optimizer()
    opt.run(n_calls=100,x0= [2.56,36.8,3.3])






