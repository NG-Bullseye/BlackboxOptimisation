from datetime import datetime
from matplotlib import pyplot as plt
from simanneal import Annealer
from skopt.utils import use_named_args
from RS_BO.Benchmark.Randomsearch import RandomSearch
from tqdm import tqdm  # import tqdm
from skopt import gp_minimize
from skopt.space import Real
import numpy as np
from RS_BO.Utility.Sim import Sim
from RS_BO.Application import Application, Sampler

class AnnealingSearch(Annealer):
    def __init__(self, state, app):
        super(AnnealingSearch, self).__init__(state)
        self.app = app
        self.fxs = []

    def move(self):
        step_size = np.random.uniform(-90, 90) * (self.T / self.Tmax)
        #if np.random.rand() > 0.5:
         #   step_size *= -90
        new_state = self.state + step_size
        new_state2=new_state
        if new_state > 90:
            new_state2 = new_state % 90
        elif new_state2 < 0:
            new_state2 = -new_state2
        if new_state2 < 0:
            print("neg")
        elif new_state2 > 90:
            print("> 90")
        print("Current step_size: ", step_size)  # Debug statement
        print("Current new_state2: ", new_state2)  # Debug statement
        self.state = new_state2

    def energy(self):
        y_value = self.app.sampler.f_discrete_real_data(np.array([self.state]))[0]
        self.fxs.append(y_value)
        return -y_value

    def anneal(self):
        self.T = self.Tmax  # Initialize T
        best_state, best_energy = super().anneal()
        return best_state, best_energy, self.fxs

    def update(self, step, T, E, acceptance, improvement):
        print("Current Temperature: ", T)  # Debug statement
        self.T = T  # Update current temperature for use in move()
        super().update(step, T, E, acceptance, improvement)

class SimAnnealSearch(RandomSearch):
    def __init__(self, app, maxiter, n_repeats):
        super().__init__(app, maxiter, n_repeats)
        self.cumulative_regret = 0.0  # Initialize cumulative regret to zero
    def anneal_search(self,Tmax,Tmin, bounds, iter):
        initial_state = np.random.uniform(*bounds)
        annealer = AnnealingSearch(initial_state, self.app)
        annealer.Tmax =Tmax
        annealer.Tmin =Tmin
        annealer.steps = iter
        state, e ,eval_points= annealer.anneal()
        eval_points = np.array(eval_points)
        go=self.app.sampler.getGlobalOptimum_Y()
        regs = np.abs(eval_points - go)
        cum_reg = np.sum(regs)
        max_found_y = -e
        return max_found_y, cum_reg,eval_points

    def calc_variance(self, max_found_y_array):
        # Convert list of arrays to 2D numpy array and compute variance along rows
        return np.var(np.array(max_found_y_array), axis=0)

    def for_range_of_iter(self, Tmax=5, Tmin=1, early_stop_threshold=100, enable_plot=False):
        total_fxs = np.zeros(self.maxiter + 1)
        total_cum_regret = np.zeros(self.maxiter + 1)
        var_fxs_array = []

        # Loop over multiple runs for statistical stability
        for run in range(self.n_repeats):
            state, e, eval_points = self.anneal_search(Tmax, Tmin, [0, 90], self.maxiter)
            # Convert eval_points to numpy array for easier manipulation
            eval_points = np.array(eval_points)
            # Fetch global optimum for regret calculation
            global_optimum = self.app.sampler.getGlobalOptimum_Y()
            # Find the best (maximum) evaluated points up to each index
            max_found_ys = np.maximum.accumulate(eval_points)
            # Append the max_found_ys array for each run to var_fxs_array
            var_fxs_array.append(max_found_ys)
            # Calculate regret for each evaluation
            regrets = np.abs(eval_points - global_optimum)
            # Calculate the cumulative regret
            cum_regret = np.cumsum(regrets)
            # Update the accumulators
            total_fxs[:len(max_found_ys)] += max_found_ys  # Fixed the slice to be up to the length of max_found_ys
            total_cum_regret += cum_regret
        # Calculate the variance for each iteration across all runs
        var_fxs = self.calc_variance(var_fxs_array)

        # Calculate the average values and variance
        avg_fxs = total_fxs / self.n_repeats
        average_cum_regret = total_cum_regret / self.n_repeats
        return avg_fxs, average_cum_regret, var_fxs

# Update the objective function
def objective(sa_search, space):
    @use_named_args(space)  # Decorator to convert a list of parameters to named arguments
    def inner_objective(**params):
        e = sa_search.for_range_of_iter(**params)[1]
        return e #[1] for best e
    return inner_objective

def run_hpOpt(sa_search):
    n_calls = 100
    space = [
        Real(1, 526167, name="Tmax"),
        Real(1, 203807, name="Tmin"),
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
    ax1.set_ylim([0, 1])
    plt.title(f"Global maximum {app.sampler.getGlobalOptimum_Y()}  n_repeats={n_repeats}")
    ax1.plot(np.array(range(0,iterations+1)), SA_fxs, color=color_list[0], label='S.A. fx')
    ax1.legend(loc='upper left')

    plt.show()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.show()
def main(app,maxiter,n_repeats):
    sa = SimAnnealSearch(app, maxiter, n_repeats)
    return sa.for_range_of_iter()

if __name__ == '__main__':
    app = Application(Sampler(Sim('Testdata')))
    iter=20
    n_repeats=11
    sa_search = SimAnnealSearch(app, iter, n_repeats)
    hpopt=False
    if hpopt:
        print(f"FINAL RESULTS SIMULATED ANNEALING: {run_hpOpt(sa_search)}")
    else:
        average_optimal_fxs, average_cumregs,var = main(app, iter, n_repeats)
        print(average_optimal_fxs)

        print(average_cumregs)
        print(f"var ")
        plot_graph_optimal_fx(app,iter,average_optimal_fxs,n_repeats)
