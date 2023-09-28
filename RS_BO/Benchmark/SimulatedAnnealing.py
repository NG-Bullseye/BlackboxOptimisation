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
from Application import Application, Sampler

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
        annealer.Tmax =Tmax#6389972 #6389972#25000.0#40.053609667034365#5.55440115545034#6.831956948536803#3#2.1195791740292957#10000#
        annealer.Tmin =Tmin#3910#1#1000000#4.273021838540448 #0.1016562260320483#0.10521097746714816#0.5#0.49614060915361824 #1000#
        annealer.steps = iter
        state, e ,eval_points= annealer.anneal()
        eval_points = np.array(eval_points)
        go=self.app.sampler.getGlobalOptimum_Y()
        regs = np.abs(eval_points - go)
        cum_reg = np.sum(regs)
        max_found_y = -e
        return max_found_y, cum_reg,eval_points

    def for_range_of_iter(self,Tmax=5  ,Tmin=1,early_stop_threshold=100, enable_plot=False):
        total_fxs = np.zeros(self.maxiter + 1)
        total_cum_regret = np.zeros(self.maxiter + 1)
        var_fxs = []
        fxs_array = np.zeros((self.n_repeats, self.maxiter + 1))

        for run in range(self.n_repeats):
            #print("next run")
            state, e, eval_points = self.anneal_search(Tmax,Tmin,[0, 90], self.maxiter)
            eval_points = np.array(eval_points)
            optima_over_iter = np.maximum.accumulate(eval_points)            # Padding
            #if len(eval_points) < self.maxiter + 1:
            #    eval_points = np.pad(eval_points, (0, self.maxiter + 1 - len(eval_points)), 'constant')

            # Storing individual runs for variance computation
            fxs_array[run, :len(eval_points)] = eval_points

            global_optimum = self.app.sampler.getGlobalOptimum_Y()
            regrets = np.abs(optima_over_iter - global_optimum)
            cum_regret = np.cumsum(regrets)

            # Accumulating fxs and regrets
            total_fxs = total_fxs[:len(optima_over_iter)] + optima_over_iter
            total_cum_regret += cum_regret

        # Average and variance calculations
        avg_fxs = total_fxs / self.n_repeats
        average_cum_regret = total_cum_regret / self.n_repeats
        var_fxs = np.var(fxs_array, axis=0)

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
