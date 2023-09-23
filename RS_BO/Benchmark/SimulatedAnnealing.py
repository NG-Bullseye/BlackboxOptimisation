from datetime import datetime

from matplotlib import pyplot as plt
from simanneal import Annealer
import numpy as np
import time
from Application import Application, Sampler
from skopt.utils import use_named_args

from Application import Sampler
from RS_BO.Benchmark.Randomsearch import RandomSearch
from RS_BO.Utility.Sim import Sim
from tqdm import tqdm  # import tqdm
from skopt import gp_minimize
from skopt.space import Real
import numpy as np
from RS_BO.Utility.Sim import Sim
from Application import Application, Sampler
import random
import time

class AnnealingSearch(Annealer):
    def __init__(self, state, app):
        super(AnnealingSearch, self).__init__(state)
        self.app = app
        self.evaluated_points = []

    def move(self):
        step_size =1.6#* (self.T / self.Tmax)# (self.T / self.Tmax)# np.random.uniform(3, 10) #  # scaling step size with T
        new_state = self.state + step_size
        self.state = np.clip(new_state, 0, 90)  # ensuring the state is within bounds

    def energy(self):
        y_value = self.app.sampler.f_discrete_real_data(np.array([self.state]))[0]
        self.evaluated_points.append(y_value)
        return -y_value

    def anneal(self):
        self.T = self.Tmax  # Initialize T
        best_state, best_energy = super().anneal()
        return best_state, best_energy, self.evaluated_points

    def update(self, step, T, E, acceptance, improvement):
        self.T = T  # Update current temperature for use in move()
        super().update(step, T, E, acceptance, improvement)

class SimAnnealSearch(RandomSearch):
    def __init__(self, app, maxiter, n_repeats):
        super().__init__(app, maxiter, n_repeats)
        self.cumulative_regret = 0.0  # Initialize cumulative regret to zero
    def anneal_search(self, bounds, iter):
        initial_state = np.random.uniform(*bounds)
        annealer = AnnealingSearch(initial_state, self.app)
        annealer.Tmax = 2.1195791740292957#10000#
        annealer.Tmin =0.49614060915361824 #1000#
        annealer.steps = iter
        state, e ,eval_points= annealer.anneal()
        eval_points = np.array(eval_points)
        go=self.app.sampler.getGlobalOptimum_Y()
        regs = np.abs(eval_points - go)
        cum_reg = np.sum(regs)
        max_found_y = -e
        return max_found_y, cum_reg
    def for_range_of_iter(self, early_stop_threshold=100, enable_plot=False):
        average_optimal_fxs = []
        average_cumregs = []
        for iter in range(1, self.maxiter + 1):
            avg_optimal_fx = 0
            avg_cum_reg = 0
            for run in range(self.n_repeats):
                max_found_y, cum_regret = self.anneal_search([0, 90], iter)
                avg_optimal_fx += max_found_y
                avg_cum_reg += cum_regret

            avg_optimal_fx /= self.n_repeats
            avg_cum_reg /= self.n_repeats
            average_optimal_fxs.append(avg_optimal_fx)
            average_cumregs.append(avg_cum_reg)
        return average_optimal_fxs, average_cumregs

    def hp_anneal_search(self, Tmax, Tmin, bounds):
        initial_state = np.random.uniform(*bounds)
        annealer = AnnealingSearch(initial_state, self.app)
        annealer.Tmax = Tmax
        annealer.Tmin = Tmin
        annealer.steps = 30
        state, e, _ = annealer.anneal()
        max_found_y = -e
        cum_reg = abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())
        return cum_reg

# Update the objective function
def objective(sa_search, space):
    @use_named_args(space)  # Decorator to convert a list of parameters to named arguments
    def inner_objective(**params):
        return sa_search.hp_anneal_search(**params, bounds=[0, 90])
    return inner_objective

def run_hpOpt(sa_search):
    n_calls = 1000
    space = [
        Real(0.5, 10.0, name="Tmax"),
        Real(0.01, 1, name="Tmin"),
        #Real(10, 500, name="iter")
    ]
    objective_with_sa_search = objective(sa_search, space)
    with tqdm(total=n_calls, desc="Optimizing", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        def callback(res):
            pbar.update(1)
        res = gp_minimize(objective_with_sa_search, space, n_calls=n_calls, acq_func="EI", n_jobs=-1, callback=callback)
    return res

def plot_graph_optimal_fx(app,iterations,SA_fxs,n_repeats,):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_list = ['b']
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel(f'Average Accuracy found over {n_repeats} repeats', color='k')
    plt.title(f"Global maximum {app.sampler.getGlobalOptimum_Y()}  n_repeats={n_repeats}")
    ax1.plot(np.array(range(0,iterations+1)), SA_fxs, color=color_list[0], label='S.A. fx')
    ax1.legend(loc='upper left')
    plt.show()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.show()
def main(app,maxiter,n_repeats):
    sa = SimAnnealSearch(app, maxiter+1, n_repeats)
    return sa.for_range_of_iter()

if __name__ == '__main__':
    app = Application(Sampler(Sim()))
    iter=30
    n_repeats=100
    sa_search = SimAnnealSearch(app, iter, n_repeats)
    hpopt=False
    if hpopt:
        print(f"FINAL RESULTS SIMULATED ANNEALING: {run_hpOpt(sa_search)}")
    else:
        average_optimal_fxs, average_cumregs = main(app, iter, n_repeats)
        print(average_optimal_fxs)
        print(average_cumregs)
        plot_graph_optimal_fx(app,iter,average_optimal_fxs,n_repeats)
