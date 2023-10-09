import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from RS_BO.Application import Application, Sampler
from RS_BO.Utility.Sim import Sim

class RandomSearch:
    def __init__(self, app,maxiter,n_repeats):
        self.app = app
        self.maxiter=maxiter
        self.n_repeats=n_repeats
        self.enable_plot = True

    def random_search(self, bounds,iter):
        lower_bound, upper_bound = bounds
        max_found_y = float('-inf')
        cum_reg=0

        for i in range(iter):
            # Single random point
            x_value = np.random.uniform(lower_bound, upper_bound)
            y_value = self.app.sampler.f_discrete_real_data(np.array([x_value]))
            y_value = y_value[0]
            max_found_y = max(max_found_y, y_value)
            cum_reg+= abs(y_value - self.app.sampler.getGlobalOptimum_Y())

        return max_found_y ,cum_reg # Just return max_found_y here

    def plot_data(self, x_values, y_values):
        plt.figure()
        plt.plot(x_values, y_values, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Percentage Close to Global Maximum')
        plt.title('Random Search Performance')
        plt.grid(True)
        plt.savefig('random_search_performance.png')
        plt.show()

    # Function to calculate variance
    def track_variance(self,max_found_y_list):
        return np.var(max_found_y_list)
    def for_range_of_iter(self, early_stop_threshold=100, enable_plot=False):
        #iter_values = []
        average_optimal_fxs=[]
        average_cumregs=[]
        var_fxs = []
        for iter in range(1,self.maxiter+1):
            avg_optimal_fx = 0
            avg_cum_reg = 0
            max_found_y_values = []
            for run in range(self.n_repeats):
                current_time = time.time()
                random.seed(int(current_time * 1e9))
                max_found_y,cum_regret = self.random_search([0, 90],iter)
                avg_optimal_fx += max_found_y  # Update it directly here
                avg_cum_reg += cum_regret
                max_found_y_values.append(max_found_y)
            var_fxs.append(self.track_variance(max_found_y_values))
            avg_optimal_fx /= self.n_repeats
            avg_cum_reg /= self.n_repeats
            average_optimal_fxs.append(avg_optimal_fx)  # Directly append to list
            average_cumregs.append(avg_cum_reg)
        return average_optimal_fxs,average_cumregs,var_fxs

    def test(self, iteration, num_executions):
        self.app = Application(Sampler(Sim("Testdata")))
        optimal_values = []
        # Execute the algorithm multiple times and store optimal_values
        for _ in range(num_executions):
            optimal_value, reg = self.random_search([0, 90], iteration)
            optimal_values.append(optimal_value)

        # Count frequency of each optimal_value
        frequency_count = defaultdict(int)
        for value in optimal_values:
            rounded_value = value  # Round to two decimal places
            frequency_count[rounded_value] += 1

        # Convert frequency count to probabilities
        probabilities = {k: v / num_executions for k, v in frequency_count.items()}

        # Sort the keys for plotting
        keys = sorted(probabilities.keys())

        # Extract the corresponding probabilities
        values = [probabilities[k] for k in keys]

        # Plotting with red crosses
        plt.scatter(keys, values, marker='x', color='red')
        plt.xlabel('Optimal Value')
        plt.ylabel('Probability')
        plt.title('Probability Distribution of Optimal Values')
        plt.show()

def main(app,maxiter,n_repeats):
    rs = RandomSearch(app, maxiter+1, n_repeats)
    return rs.for_range_of_iter()

if __name__ == '__main__':
    app = Application(Sampler(Sim("Testdata")))
    rs = RandomSearch(app, 0 + 1, 1)
    rs.test(iteration = 1,num_executions = 1000)

