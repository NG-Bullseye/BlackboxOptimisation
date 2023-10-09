import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate
from RS_BO.Application import Application, Sampler
from RS_BO.Utility.Sim import Sim


def parse_filename(filename):
    match = re.search(r'n_repeats_(\d+)_iterations_(\d+)', filename.split("/")[-1])
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        print("Filename format is incorrect.")
        return None


class PlottingResults:
    def __init__(self, csv_path_regrets, csv_path_optima):
        self.csv_path_regrets = csv_path_regrets
        self.csv_path_optima = csv_path_optima
        self.scale = 1
        self.load_csv_data()

        self.app = Application(Sampler(Sim()))
        self.yaw_acc = self.app.sampler.yaw_acc

        n_repeats_regret, iterations_regret = parse_filename(self.csv_path_regrets)
        n_repeats_opt, iterations_opt = parse_filename(self.csv_path_optima)

        self.plot_regret_graph(n_repeats_regret, iterations_regret)
        self.plot_optimal_fx_graph(n_repeats_opt, iterations_opt)
        self.plot_yaw_accuracy()

    def load_csv_data(self):
        self.data_regrets = pd.read_csv(self.csv_path_regrets)
        self.data_optima = pd.read_csv(self.csv_path_optima)

    def exponential_moving_average(self,y, alpha=0.3):
        """
        Applies Exponential Moving Average (EMA) on a series
        :param y: the input series
        :param alpha: smoothing factor, between 0 and 1
        :return: smoothed series
        """
        s = [y[0]]
        for t in range(1, len(y)):
            s.append(alpha * y[t] + (1 - alpha) * s[t - 1])
        return np.array(s)
    def plot_graph(self, ax, x, y, label, color, smooth=False, alpha=0.3):
        ax.set_ylim([0.7, 0.95])
        if smooth:
            y_smoothed = self.exponential_moving_average(y, alpha)
            ax.plot(x, y_smoothed, color=color, label=label + ' (Smoothed)')
        else:
            ax.plot(x, y, color=color, label=label)

    def plot_regret_graph(self, n_repeats, iterations):
        fig, ax = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel(f'Average Cumulative Regrets over {n_repeats} repeats')

        for label, color in zip(self.data_regrets.columns[1:], ['b', 'g', 'r', 'm', 'c']):
            self.plot_graph(ax, range(iterations), self.data_regrets[label], label, color, smooth=True)

        self.save_plot(f"Regrets_{datetime.now().strftime('%Y%m%d')}", 'cumulative_regrets_plot.png', plt)
        ax.legend(loc='upper left')
        plt.show()

    def plot_optimal_fx_graph(self, n_repeats, iterations):
        fig, ax = plt.subplots(figsize=(10 * self.scale, 6 * self.scale))
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel(f'Accuracy found over {n_repeats} Repeats')

        for label, color in zip(self.data_optima.columns[1:], ['b', 'g', 'r', 'm', 'c', 'y']):
            self.plot_graph(ax, range(iterations), self.data_optima[label], label, color, smooth=True)

        self.save_plot(f"Graphs_{datetime.now().strftime('%Y%m%d')}", 'optimal_fxs_plot.png', plt)
        ax.legend(loc='upper left')
        plt.show()

    def save_plot(self, folder_name, plot_name, plt_obj):
        path = os.path.join("plots", folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        plt_obj.savefig(os.path.join(path, plot_name))

    def plot_yaw_accuracy(self):
        yaw, accuracy = list(self.yaw_acc.keys()), list(self.yaw_acc.values())
        f = interpolate.interp1d(yaw, accuracy, kind='cubic')
        yaw_new = np.linspace(min(yaw), max(yaw), 1000)
        accuracy_new = f(yaw_new)

        plt.scatter(yaw, accuracy, label='Sampled Data', color='r')
        plt.plot(yaw_new, accuracy_new, label='Interpolated Data', linestyle='--')
        plt.xlabel('Yaw')
        plt.ylabel('Measured Accuracy')
        plt.legend()
        self.save_plot(f"objfunc_{datetime.now().strftime('%Y%m%d')}", 'CNN_Performance_of_Camera_Positions.png', plt)
        plt.show()
PlottingResults(
    "/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/Regrets_PlottingData_n_repeats_2_iterations_11.csv"
    ,
    "/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/OptimalFX_PlottingData_n_repeats_2_iterations_11.csv"
)