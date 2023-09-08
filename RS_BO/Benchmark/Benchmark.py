import os
from datetime import datetime
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
from Randomsearch import main as RSmain
from BaysianOptimization import main as BOmain
from Gridsearch import main as GSmain
from Custom_Baysian_Optimisation import main as CBOmain
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, scale=1, maxiter=1, n_repeats=1):
        self.app = Application(Sampler(Sim()))
        self.maxiter = maxiter
        self.n_repeats = n_repeats
        self.scale = scale

        iterations = list(range(0, self.maxiter + 1))

        RS_avg_optimal_fxs, RS_avg_cum_regrets = RSmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS RANDOMSEARCH: \navg_optimal_fxs: {RS_avg_optimal_fxs} \navg_cum_regrets:{RS_avg_cum_regrets}")

        BO_avg_optimal_fxs, BO_avg_cum_regrets = BOmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS VANILLA BO: \navg_optimal_fxs: {BO_avg_optimal_fxs} \navg_cum_regrets:{BO_avg_cum_regrets}")

        GS_avg_optimal_fxs, GS_avg_cum_regrets = GSmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS GRIDSEARCH: \navg_optimal_fxs: {GS_avg_optimal_fxs} \navg_cum_regrets:{GS_avg_cum_regrets}")

        CBO_avg_optimal_fxs, CBO_avg_cum_regrets = CBOmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {CBO_avg_optimal_fxs} \navg_cum_regrets:{CBO_avg_cum_regrets}")

        self.plot_graph(iterations, GS_avg_optimal_fxs, BO_avg_optimal_fxs, CBO_avg_optimal_fxs, RS_avg_optimal_fxs)
        self.plot_regret_graph(iterations, GS_avg_cum_regrets, BO_avg_cum_regrets, CBO_avg_cum_regrets,RS_avg_cum_regrets)

    def plot_graph(self, iterations, GS_fxs, BO_fxs, CBO_fxs, RS_fxs):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        color_list = ['b', 'g', 'r', 'm']
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel('Average Percentage Close to Global Maximum', color='k')

        ax1.plot(iterations, GS_fxs, color=color_list[0], label='GS_fxs')
        ax1.plot(iterations, BO_fxs, color=color_list[1], label='BO_fxs')
        ax1.plot(iterations, CBO_fxs, color=color_list[2], label='CBO_fxs')
        ax1.plot(iterations, RS_fxs, color=color_list[3], label='RS_fxs')

        ax1.legend(loc='upper left')
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Graphs_{timestamp}"
        self.save_plot(folder_name, 'optimal_fxs_plot.png')
        plt.show()

    def plot_regret_graph(self, iterations, GS_regrets, BO_regrets, CBO_regrets, RS_regrets):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        color_list = ['b', 'g', 'r', 'm']
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel('Average Cumulative Regrets', color='k')

        ax1.plot(iterations, GS_regrets, color=color_list[0], label='GS_regrets')
        ax1.plot(iterations, BO_regrets, color=color_list[1], label='BO_regrets')
        ax1.plot(iterations, CBO_regrets, color=color_list[2], label='CBO_regrets')
        ax1.plot(iterations, RS_regrets, color=color_list[3], label='RS_regrets')

        ax1.legend(loc='upper left')
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Regrets_{timestamp}"
        self.save_plot(folder_name, 'cumulative_regrets_plot.png')
        plt.show()

    def save_plot(self,folder_name, file_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f'{folder_name}/{file_name}')

if __name__ == "__main__":
    benchmark = Benchmark(scale=1, maxiter=10, n_repeats=10)
