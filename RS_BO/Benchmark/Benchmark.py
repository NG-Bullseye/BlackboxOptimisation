import os
from RS_BO.Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
from Randomsearch import main as RSmain
from BaysianOptimization import main as BOmain
from Gridsearch import main as GSmain
from Custom_Baysian_Optimisation import main as CBOmain
from Custom_Baysian_Optimisation import main_no_rec as CBO_no_rec_main
import pandas as pd
from SimulatedAnnealing import main as SAmain
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, scale=1, maxiter=1, n_repeats=1):
        self.color_map = {
            'RandomSearch': 'c',
            'SimulatedAnnealing': 'y',
            'BayesianOptimization': 'g',
            'GridSearch': 'b',
            'XAI': 'r',
            'BO': 'm'
        }

        self.label_map = {
            'RandomSearch': 'Randomsearch',
            'SimulatedAnnealing': 'Simulated Annealing',
            'BayesianOptimization': 'Baysianopt',
            'GridSearch': 'Gridsearch',
            'XAI': 'BO with XAI',
            'BO': 'BO without XAI'
        }
        self.app = Application(Sampler(Sim("Testdata")))
        self.maxiter = maxiter
        self.n_repeats = n_repeats
        self.scale = scale

    def run(self):
        iterations = list(range(0, self.maxiter + 1))

        RS_avg_optimal_fxs, RS_avg_cum_regrets, RS_var_fxs = RSmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS RANDOMSEARCH: \navg_optimal_fxs: {RS_avg_optimal_fxs} \navg_cum_regrets:{RS_avg_cum_regrets}")

        SA_avg_optimal_fxs, SA_avg_cum_regrets, SA_var_fxs = SAmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS RANDOMSEARCH: \navg_optimal_fxs: {SA_avg_optimal_fxs} \navg_cum_regrets:{SA_avg_cum_regrets}")

        BO_avg_optimal_fxs, BO_avg_cum_regrets, BO_var_fxs = BOmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS VANILLA BO: \navg_optimal_fxs: {BO_avg_optimal_fxs} \navg_cum_regrets:{BO_avg_cum_regrets}")

        GS_avg_optimal_fxs, GS_avg_cum_regrets, GS_var_fxs = GSmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS GRIDSEARCH: \navg_optimal_fxs: {GS_avg_optimal_fxs} \navg_cum_regrets:{GS_avg_cum_regrets}")

        CBO_avg_optimal_fxs, CBO_avg_cum_regrets, CBO_var_fxs = CBOmain(self.app, self.maxiter, self.n_repeats)
        print(
            f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {CBO_avg_optimal_fxs} \navg_cum_regrets:{CBO_avg_cum_regrets}")

        CBO_no_rec_avg_optimal_fxs, CBO_no_rec_avg_cum_regrets, CBO_no_rec_var_fxs = CBO_no_rec_main(self.app,
                                                                                                     self.maxiter,
                                                                                                     self.n_repeats)
        print(
            f"FINAL RESULTS CUSTOM BO: \navg_optimal_fxs: {CBO_no_rec_avg_optimal_fxs} \navg_cum_regrets:{{CBO_no_rec_avg_cum_regrets}}")
        # Sample usage



        self.store_variance_data_into_csv_file(iterations,
                                               GS_var_fxs,
                                               BO_var_fxs,
                                               CBO_var_fxs,
                                               CBO_no_rec_var_fxs,
                                               RS_var_fxs,
                                               SA_var_fxs)

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

        self.plot_graph_optimal_fx_var(iterations,
                                       RandomSearch=RS_var_fxs,
                                       SimulatedAnnealing=SA_var_fxs,
                                       BayesianOptimization=BO_var_fxs,
                                       GridSearch=GS_var_fxs,
                                       XAI=CBO_var_fxs,
                                       BO=CBO_no_rec_var_fxs)
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

    def store_variance_data_into_csv_file(self, iterations, GS_var_fxs, BO_var_fxs, CBO_var_fxs, CBO_no_rec_var_fxs,
                                          RS_var_fxs, SA_var_fxs):
        # Create a Pandas DataFrame
        df = pd.DataFrame({
            'Iterations': iterations,
            'Gridsearch_VarianceFX': GS_var_fxs,
            'Baysianopt_VarianceFX': BO_var_fxs,
            'AI_Search_VarianceFX': CBO_var_fxs,
            'My_BO_VarianceFX': CBO_no_rec_var_fxs,
            'Randomsearch_VarianceFX': RS_var_fxs,
            'Sim_An_VarianceFX': SA_var_fxs
        })

        # Generate filename with timestamp, n_repeats, and iterations
        filename = f"VarianceFX_PlottingData_n_repeats_{self.n_repeats}_iterations_{len(iterations)}.csv"

        # Save DataFrame to CSV
        df.to_csv(filename, index=False)
        print(f"Variance data stored in {filename}")

    def plot_graph_optimal_fx_var(self, iterations, **kwargs):
        plt.figure(figsize=(12, 8))

        for algo_name, var_fxs in kwargs.items():
            plt.plot(iterations, var_fxs, label=self.label_map.get(algo_name, algo_name),
                     color=self.color_map.get(algo_name, 'k'))

        plt.xlabel('Iterations [count]')
        plt.ylabel("Variance of f(x) [unitless]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_regret_graph(self, iterations, GS_regrets, BO_regrets, CBO_regrets, CBO_no_rec, RS_regrets,
                          SA_avg_optimal_fxs):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        ax1.set_xlabel('Number of Iterations [count]')
        ax1.set_ylabel(f'Average Cumulative Regrets over {self.n_repeats} repeats [unitless]', color='k')

        ax1.plot(iterations, GS_regrets, color=self.color_map['GridSearch'], label='Gridsearch')
        ax1.plot(iterations, BO_regrets, color=self.color_map['BayesianOptimization'], label='Baysianopt')
        ax1.plot(iterations, CBO_regrets, color=self.color_map['XAI'], label='BO with XAI')
        ax1.plot(iterations, CBO_no_rec, color=self.color_map['BO'], label='BO without XAI')
        ax1.plot(iterations, RS_regrets, color=self.color_map['RandomSearch'], label='Randomsearch')
        ax1.plot(iterations, SA_avg_optimal_fxs, color=self.color_map['SimulatedAnnealing'], label='Simulated Annealing')

        ax1.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_graph_optimal_fx(self, iterations, GS_fxs, BO_fxs, CBO_fxs, CBO_no_recs_fxs, RS_fxs, SA_fxs):
        fig, ax1 = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))

        color_list = ['b', 'g', 'r', 'm', 'c', 'y']
        ax1.set_xlabel('Number of Iterations [count]')
        ax1.set_ylabel(f'Accuracy [unitless]', color='k')

        ax1.plot(iterations, GS_fxs, color=color_list[0], label='Gridsearch')
        ax1.plot(iterations, BO_fxs, color=color_list[1], label='Baysianopt')
        ax1.plot(iterations, CBO_fxs, color=color_list[2], label='BO with XAI')
        ax1.plot(iterations, CBO_no_recs_fxs, color=color_list[3], label='BO without XAI')
        ax1.plot(iterations, RS_fxs, color=color_list[4], label='Randomsearch')
        ax1.plot(iterations, SA_fxs, color=color_list[5], label='Simulated Annealing')

        ax1.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
    def save_plot(self,folder_name, file_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f'{folder_name}/{file_name}')

    def load_plots(self, regrets_csv_path=None, optimalfx_csv_path=None, var_csv_path=None):
        # Load data from CSV files

        # Extract iteration numbers

        # Plot the regret graph
        if regrets_csv_path is not None:
            df_regrets = pd.read_csv(regrets_csv_path)
            iterations = df_regrets['Iterations']
            self.plot_regret_graph(iterations,
                                   df_regrets['Gridsearch_Regrets'],
                                   df_regrets['Baysianopt_Regrets'],
                                   df_regrets['AI_Search_Regrets'],
                                   df_regrets['My_BO_Regrets'],
                                   df_regrets['Randomsearch_Regrets'],
                                   df_regrets['Sim_An_Regrets'])

        if optimalfx_csv_path is not None:# Plot the optimal f(x) graph
            df_optimalfx = pd.read_csv(optimalfx_csv_path)
            iterations = df_optimalfx['Iterations']
            self.plot_graph_optimal_fx(iterations,
                                       df_optimalfx['Gridsearch_OptimalFX'],
                                       df_optimalfx['Baysianopt_OptimalFX'],
                                       df_optimalfx['AI_Search_OptimalFX'],
                                       df_optimalfx['My_BO_OptimalFX'],
                                       df_optimalfx['Randomsearch_OptimalFX'],
                                       df_optimalfx['Sim_An_OptimalFX'])

        if var_csv_path is not None: # Plot the variance graph
            df_var = pd.read_csv(var_csv_path)
            iterations = df_var['Iterations']
            self.plot_graph_optimal_fx_var(iterations,
                                           RandomSearch=df_var['Randomsearch_VarianceFX'],
                                           SimulatedAnnealing=df_var['Sim_An_VarianceFX'],
                                           BayesianOptimization=df_var['Baysianopt_VarianceFX'],
                                           GridSearch=df_var['Gridsearch_VarianceFX'],
                                           XAI=df_var['AI_Search_VarianceFX'],
                                           BO=df_var['My_BO_VarianceFX'])

if __name__ == "__main__":
    benchmark = Benchmark(scale=1, maxiter=20, n_repeats=300)
    run=True
    plot = True
    if run:
       benchmark.run() #saves as csv
    if plot: #loads csv manualy
        benchmark.load_plots(
            optimalfx_csv_path="/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/OptimalFX_PlottingData_n_repeats_300_iterations_21.csv",
            regrets_csv_path="/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/Regrets_PlottingData_n_repeats_300_iterations_21.csv",
            var_csv_path="/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/VarianceFX_PlottingData_n_repeats_300_iterations_21.csv"
        )
