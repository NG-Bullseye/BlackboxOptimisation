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
        step_size = np.random.uniform(-1, 1) * (self.T / self.Tmax)
        if np.random.rand() > 0.5:
            step_size *= -1
        new_state = self.state + step_size
        if new_state < 0:
            new_state = -new_state
        elif new_state > 90:
            new_state = 180 - new_state
        self.state = new_state

    def energy(self):
        y_value = self.app.sampler.f_discrete_real_data(np.array([self.state]))[0]
        self.evaluated_points.append(y_value)
        return -y_value

    def anneal(self):
        self.T = self.Tmax  # Initialize T
        best_state, best_energy = super().anneal()
        return best_state, best_energy, self.evaluated_points

    def update(self, step, T, E, acceptance, improvement):
        print("Current Temperature: ", T)  # Debug statement
        self.T = T  # Update current temperature for use in move()
        super().update(step, T, E, acceptance, improvement)

class SimAnnealSearch(RandomSearch):
    def __init__(self, app, maxiter, n_repeats):
        super().__init__(app, maxiter, n_repeats)
        self.cumulative_regret = 0.0  # Initialize cumulative regret to zero
    def anneal_search(self,Tmax, Tmin,bounds, iter):
        initial_state = np.random.uniform(*bounds)
        annealer = AnnealingSearch(initial_state, self.app)
        annealer.Tmax =Tmax #689.7648753660962#40.053609667034365#5.55440115545034#6.831956948536803#3#2.1195791740292957#10000#
        annealer.Tmin =Tmin#61.32428326530278#4.273021838540448 #0.1016562260320483#0.10521097746714816#0.5#0.49614060915361824 #1000#
        annealer.steps = iter
        state, e ,eval_points= annealer.anneal()
        eval_points = np.array(eval_points)
        go=self.app.sampler.getGlobalOptimum_Y()
        regs = np.abs(eval_points - go)
        cum_reg = np.sum(regs)
        max_found_y = -e
        return max_found_y, cum_reg

    def for_range_of_iter(self, Tmax=689689,Tmin=689, early_stop_threshold=100, enable_plot=False):
        average_optimal_fxs = []
        average_cumregs = []
        var_fxs = []
        for iter in range(1, self.maxiter + 1):

            avg_optimal_fx = 0
            avg_cum_reg = 0
            max_found_y_values = []
            for run in range(self.n_repeats):
                print("next run")
                max_found_y, cum_regret = self.anneal_search(Tmax,Tmin,[0, 90], iter)
                avg_optimal_fx += max_found_y
                avg_cum_reg += cum_regret
                max_found_y_values.append(max_found_y)
            avg_optimal_fx /= self.n_repeats
            avg_cum_reg /= self.n_repeats
            average_optimal_fxs.append(avg_optimal_fx)
            average_cumregs.append(avg_cum_reg)
        return average_optimal_fxs, average_cumregs,var_fxs

    def hp_anneal_search(self, Tmax, Tmin, bounds):
        initial_state = np.random.uniform(*bounds)
        annealer = AnnealingSearch(initial_state, self.app)
        annealer.Tmax = Tmax
        annealer.Tmin = Tmin
        annealer.steps = 20
        state, e, _ = annealer.anneal()
        max_found_y = -e
        cum_reg = abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())
        return max_found_y

# Update the objective function
def objective(sa_search, space):
    @use_named_args(space)  # Decorator to convert a list of parameters to named arguments
    def inner_objective(**params):
        return sa_search.for_range_of_iter(**params, bounds=[0, 90])
    return inner_objective

def run_hpOpt(sa_search):
    n_calls = 300
    space = [
        Real(1, 1000.0, name="Tmax"),
        Real(0.1, 100, name="Tmin"),
        #Real(1, 10, name="step")
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
    app = Application(Sampler(Sim('Testdata')))
    iter=20
    n_repeats=2001
    sa_search = SimAnnealSearch(app, iter, n_repeats)
    hpopt=False
    if hpopt:
        print(f"FINAL RESULTS SIMULATED ANNEALING: {run_hpOpt(sa_search)}")
    else:
        average_optimal_fxs, average_cumregs,var = main(app, iter, n_repeats)
        print(average_optimal_fxs)

        print(average_cumregs)
        print(f"var {var}")
        plot_graph_optimal_fx(app,iter,average_optimal_fxs,n_repeats)
