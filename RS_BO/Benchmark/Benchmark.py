import os
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
from Randomsearch import main as RSmain
from BaysianOptimization import main as BOmain
from Gridsearch import main as GSmain
from Custom_Baysian_Optimisation import main as CBOmain
from Custom_Baysian_Optimisation import main_no_rec as CBO_no_rec_main
import pandas as pd
from SimulatedAnnealing import main as SAmain
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
class Benchmark:
    def __init__(self, scale=1, maxiter=1, n_repeats=1):
        self.app = Application(Sampler(Sim()))
        self.maxiter = maxiter
        self.n_repeats = n_repeats
        self.scale = scale

        iterations = list(range(0, self.maxiter + 1))

        RS_avg_optimal_fxs, RS_avg_cum_regrets = RSmain(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS RANDOMSEARCH: \navg_optimal_fxs: {RS_avg_optimal_fxs} \navg_cum_regrets:{RS_avg_cum_regrets}")

        SA_avg_optimal_fxs, SA_avg_cum_regrets = SAmain(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS RANDOMSEARCH: \navg_optimal_fxs: {SA_avg_optimal_fxs} \navg_cum_regrets:{SA_avg_cum_regrets}")

        BO_avg_optimal_fxs, BO_avg_cum_regrets = BOmain(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS VANILLA BO: \navg_optimal_fxs: {BO_avg_optimal_fxs} \navg_cum_regrets:{BO_avg_cum_regrets}")

        GS_avg_optimal_fxs, GS_avg_cum_regrets = GSmain(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS GRIDSEARCH: \navg_optimal_fxs: {GS_avg_optimal_fxs} \navg_cum_regrets:{GS_avg_cum_regrets}")

        CBO_avg_optimal_fxs, CBO_avg_cum_regrets = CBOmain(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {CBO_avg_optimal_fxs} \navg_cum_regrets:{CBO_avg_cum_regrets}")

        CBO_no_rec_avg_optimal_fxs, CBO_no_rec_avg_cum_regrets = CBO_no_rec_main(self.app, self.maxiter, self.n_repeats)
        print(f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {CBO_no_rec_avg_optimal_fxs} \navg_cum_regrets:{{CBO_no_rec_avg_cum_regrets}}")
        self.plot_regret_graph(iterations,
                               GS_avg_cum_regrets,
                               BO_avg_cum_regrets,
                               CBO_avg_cum_regrets,
                               CBO_no_rec_avg_cum_regrets,
                               RS_avg_cum_regrets,
                               SA_avg_cum_regrets)
        self.plot_graph_optimal_fx(iterations,
                                   GS_avg_optimal_fxs,
                                   BO_avg_optimal_fxs,
                                   CBO_avg_optimal_fxs,
                                   CBO_no_rec_avg_optimal_fxs,
                                   RS_avg_optimal_fxs,
                                   SA_avg_optimal_fxs)

        self.store_plottingdata_into_csv_file_regrets(iterations,
                                                      GS_avg_cum_regrets,
                                                      BO_avg_cum_regrets,
                                                      CBO_avg_cum_regrets,
                                                      CBO_no_rec_avg_cum_regrets,
                                                      RS_avg_cum_regrets,
                                                      SA_avg_cum_regrets)
        self.store_plottingdata_into_csv_file_optimalfx(iterations,
                                                        GS_avg_optimal_fxs,
                                                        BO_avg_optimal_fxs,
                                                        CBO_avg_optimal_fxs,
                                                        CBO_no_rec_avg_optimal_fxs,
                                                        RS_avg_optimal_fxs,
                                                        SA_avg_optimal_fxs)

        #self.plot_graph_optimal_fx(iterations, GS_avg_optimal_fxs, BO_avg_optimal_fxs, CBO_avg_optimal_fxs,CBO_no_rec_avg_optimal_fxs, RS_avg_optimal_fxs,SA_avg_optimal_fxs)

        #self.plot_graph_optimal_fx_interp(iterations, GS_avg_optimal_fxs, BO_avg_optimal_fxs, CBO_avg_optimal_fxs, RS_avg_optimal_fxs)
        #self.plot_graph_optimal_fx_regres(iterations, GS_avg_optimal_fxs, BO_avg_optimal_fxs, CBO_avg_optimal_fxs, RS_avg_optimal_fxs)
        #self.plot_optimal_fx_smooth(iterations, GS_avg_optimal_fxs, BO_avg_optimal_fxs, CBO_avg_optimal_fxs,CBO_no_rec_avg_optimal_fxs, RS_avg_optimal_fxs)
        #self.plot_regret_regres(iterations, GS_avg_cum_regrets, BO_avg_cum_regrets, CBO_avg_cum_regrets,RS_avg_cum_regrets)
        #self.plot_regret_interp(iterations, GS_avg_cum_regrets, BO_avg_cum_regrets, CBO_avg_cum_regrets,RS_avg_cum_regrets)



    def store_plottingdata_into_csv_file_regrets(self, iterations, GS_regrets, BO_regrets, CBO_regrets, CBO_no_rec,
                                                 RS_regrets, SA_regrets):
        # Create a Pandas DataFrame
        df = pd.DataFrame({
            'Iterations': iterations,
            'Gridsearch_Regrets': GS_regrets,
            'Baysianopt_Regrets': BO_regrets,
            'AI_Search_Regrets': CBO_regrets,
            'My_BO_Regrets': CBO_no_rec,
            'Randomsearch_Regrets': RS_regrets,
            'Sim_An_Regrets': SA_regrets
        })

        # Generate filename with timestamp, n_repeats, and iterations
        filename = f"Regrets_PlottingData_n_repeats_{self.n_repeats}_iterations_{len(iterations)}.csv"

        # Save DataFrame to CSV
        df.to_csv(filename, index=False)
        print(f"Data stored in {filename}")

    def store_plottingdata_into_csv_file_optimalfx(self, iterations, GS_fxs, BO_fxs, CBO_fxs, CBO_no_rec, RS_fxs,
                                                   SA_fxs):
        # Create a Pandas DataFrame
        df = pd.DataFrame({
            'Iterations': iterations,
            'Gridsearch_OptimalFX': GS_fxs,
            'Baysianopt_OptimalFX': BO_fxs,
            'AI_Search_OptimalFX': CBO_fxs,
            'My_BO_OptimalFX': CBO_no_rec,
            'Randomsearch_OptimalFX': RS_fxs,
            'Sim_An_OptimalFX': SA_fxs
        })

        # Generate filename with timestamp, n_repeats, and iterations
        filename = f"OptimalFX_PlottingData_n_repeats_{self.n_repeats}_iterations_{len(iterations)}.csv"

        # Save DataFrame to CSV
        df.to_csv(filename, index=False)
        print(f"Data stored in {filename}")

    def plot_regret_graph(self, iterations, GS_regrets, BO_regrets, CBO_regrets, CBO_no_rec, RS_regrets,
                          SA_avg_optimal_fxs):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        color_list = ['b', 'g', 'r', 'm', 'c', 'y']
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel(f'Average Cumulative Regrets over {self.n_repeats} repeats', color='k')

        ax1.plot(iterations, GS_regrets, color=color_list[0], label='GS_regrets')
        ax1.plot(iterations, BO_regrets, color=color_list[1], label='BO_regrets')
        ax1.plot(iterations, CBO_regrets, color=color_list[2], label='CBO_regrets')
        ax1.plot(iterations, CBO_no_rec, color=color_list[3], label='CBO_no_recs_regrets')
        ax1.plot(iterations, RS_regrets, color=color_list[4], label='RS_regrets')
        ax1.plot(iterations, SA_avg_optimal_fxs, color=color_list[5], label='SA_avg_optimal_fxs')

        plt.title(f"Cumulative regret")

        ax1.legend(loc='upper left')
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Regrets_{timestamp}"
        self.save_plot(folder_name, 'cumulative_regrets_plot.png')
        plt.show()

    def plot_graph_optimal_fx(self, iterations, GS_fxs, BO_fxs, CBO_fxs, CBO_no_recs_fxs, RS_fxs, SA_fxs):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        color_list = ['b', 'g', 'r', 'm', 'c', 'y']
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel(f'Average Accuracy found over {self.n_repeats} repeats', color='k')
        plt.title(f"Global maximum {self.app.sampler.getGlobalOptimum_Y()}  n_repeats={self.n_repeats}")
        ax1.plot(iterations, GS_fxs, color=color_list[0], label='Gridsearch')
        ax1.plot(iterations, BO_fxs, color=color_list[1], label='Baysianopt')
        ax1.plot(iterations, CBO_fxs, color=color_list[2], label='A.I. Search')
        ax1.plot(iterations, CBO_no_recs_fxs, color=color_list[3], label='Search')
        ax1.plot(iterations, RS_fxs, color=color_list[4], label='Randomsearch')
        ax1.plot(iterations, SA_fxs, color=color_list[5], label='S.A. fx')

        ax1.legend(loc='upper left')
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Graphs_{timestamp}"
        self.save_plot(folder_name, 'optimal_fxs_plot.png')
        plt.show()

    def save_plot(self,folder_name, file_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f'{folder_name}/{file_name}')

if __name__ == "__main__":
    benchmark = Benchmark(scale=1, maxiter=10, n_repeats=2)
