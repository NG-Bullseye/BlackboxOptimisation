import numpy as np
import random
import time
import matplotlib.pyplot as plt
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
import random
import time

class RandomSearch:
    def __init__(self, app,maxiter,n_repeats):
        self.app = app
        self.maxiter=maxiter
        self.n_repeats=n_repeats
        self.early_stop_threshold = 95
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
            cum_reg+= abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())
            #print(f"iter{iter} y_value{y_value} max_found_y{max_found_y}  i{i} cum_reg{cum_reg} reg{abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())}")

            #print(f"Single iteration max_found_y: {max_found_y}")

            #global_max = self.app.sampler.getGlobalOptimum_Y()
            #percentage_close = (max_found_y / global_max) * 100

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

    def for_range_of_iter(self, early_stop_threshold=100, enable_plot=False):
        #iter_values = []
        average_optimal_fxs=[]
        average_cumregs=[]

        for iter in range(1,self.maxiter+1):
            avg_optimal_fx = 0
            avg_cum_reg = 0
            for run in range(self.n_repeats):
                current_time = time.time()
                random.seed(int(current_time * 1e9))
                max_found_y,cum_regret = self.random_search([0, 90],iter)
                avg_optimal_fx += max_found_y  # Update it directly here
                avg_cum_reg += cum_regret
            avg_optimal_fx /= self.n_repeats
            avg_cum_reg /= self.n_repeats
            average_optimal_fxs.append(avg_optimal_fx)  # Directly append to list
            average_cumregs.append(avg_cum_reg)

        #if enable_plot:
            #self.plot_data([i[0] for i in average_optimal_fxs], [i[1] for i in average_optimal_fxs])
        return average_optimal_fxs,average_cumregs




# Example usage


def main(app,maxiter,n_repeats):
    rs = RandomSearch(app, maxiter+1, n_repeats)
    return rs.for_range_of_iter()

if __name__ == '__main__':
    app = Application(Sampler(Sim()))
    print(f"FINAL RESULTS RANDOMSEARCH: {main(app,0,1111)}")
