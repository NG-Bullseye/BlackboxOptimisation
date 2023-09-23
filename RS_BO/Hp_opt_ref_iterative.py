from tqdm import tqdm
from skopt import gp_minimize
from RS_BO.Utility.Sim import Sim
from Application import Application, Sampler
import random
import time
import numpy as np

class Optimizer:
    def __init__(self):
        self.data = Sampler(Sim())
        self.last_best_params = None
        self.original_bounds = [
            #[0.005, 20.0] #kernel_scale
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
        offset_scale, offset_range, protection_width = params#kernel_scale= params#,
        montecarlo_iter = 3
        average_cumulative_regret_sum = 0.
        for i in range(montecarlo_iter):
            current_time = time.perf_counter()
            randomseed = random.seed(int(current_time * 1e9))
            app = Application(self.data)
            cumulative_regret = app.start_sim_with_real_data(1,
                                                             kernel_scale=2.221461291655982,
                                                             offset_scale=offset_scale,#0.31899966392534623, 0.6953434383380036, ,
                                                             offset_range=offset_range,#37.15319225574285, 50.92663911701745
                                                             protection_width=protection_width,#0.5809482476332607, 3.2715701918611297
                                                             n_iterations=10,
                                                             randomseed=randomseed)
            average_cumulative_regret_sum += cumulative_regret
        average_cumulative_regret = average_cumulative_regret_sum / montecarlo_iter
        return average_cumulative_regret

    def run(self, n_calls, bounds=None, x0=None):
        converged = False
        while not converged:
            bounds = self.original_bounds.copy()  # Add this before while loop
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
    opt.run(n_calls=100,x0= [2.56,36.8,3.3])#, 0.18080060285014135, 24.979610423583395, 0.8845792950045508])

#2.221461291655982    lange iterative kernel analyse
#3.4179120262940454
#[1.0302249999999997, 0.01, 20.0, 0.8974170487797348]

#x0: [5.1648468262417735, 0.31899966392534623, 37.15319225574285, 0.5809482476332607],
#x0: [1.0831035714270065, 3.1757570319619477, 4.2399380234330515, 0.06432944199790688],
#x0: [3.4179120262940454, 2.847880535002211, 49.063651445207455, 4.360958771609868],
#x0: [3.244589455768782, 2.9397881688176244, 49.93341453273596, 4.173637658673011],
#x0: [4.977806753868584, 0.9712186246175858, 49.04501685861087, 0.5380242395825713],
#x0: [3.5717598620426267, 3.9297508966880867, 45.720586189996986, 3.6935558712633725],
#x0: [3.5436200680418017, 3.740383982202573, 45.6695461058799, 3.831339286347537],
#x0: [3.9936879722641976, 2.617022740460637, 13.594953784639179, 9.201089454571893],
#
#
#Best parameters: [1.0, 0.8112797106022619, 23.026629235156467, 0.2635513742728467] min_cumulative_regret_global: inf




