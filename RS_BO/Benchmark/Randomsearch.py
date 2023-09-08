import numpy as np
import random
import time
import matplotlib.pyplot as plt
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class RandomSearch:
    def __init__(self, app,maxiter,n_repeats):
        self.app = app
        self.maxiter=maxiter
        self.n_repeats=n_repeats
    def random_search(self, bounds):
        lower_bound, upper_bound = bounds
        max_found_y = float('-inf')
        random_points = np.random.uniform(lower_bound, upper_bound, self.maxiter)

        for i, x_value in enumerate(random_points):
            y_value = self.app.sampler.f_discrete_real_data(np.array([x_value]))
            y_value = y_value[0]
            max_found_y = max(max_found_y, y_value)
            print(f"Iterer: {i} max_found_y: {max_found_y}")

        global_max = self.app.sampler.getGlobalOptimum_Y()
        percentage_close = (max_found_y / global_max) * 100
        return percentage_close

    def plot_data(self, x_values, y_values):
        plt.figure()
        plt.plot(x_values, y_values, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Percentage Close to Global Maximum')
        plt.title('Random Search Performance')
        plt.grid(True)
        plt.savefig('random_search_performance.png')
        plt.show()

    def for_range_of_iter(self, early_stop_threshold=0, enable_plot=False):
        iter_values = []

        for iter in range(0, self.maxiter + 1):
            avg_percentage_close = 0

            for run in range(self.n_repeats):
                current_time = time.time()
                random.seed(int(current_time * 1e9))
                percentage_close = self.random_search([0, 90])
                avg_percentage_close += percentage_close

            avg_percentage_close /= self.n_repeats
            print(f"For num_points = {iter}, Average Percentage Close: {avg_percentage_close}%")

            iter_values.append((iter, avg_percentage_close))

            if avg_percentage_close >= early_stop_threshold:
                print("Stopping early.")
                break

        if enable_plot:
            self.plot_data([i[0] for i in iter_values], [i[1] for i in iter_values])
        return iter_values


# Example usage

early_stop_threshold = 95
enable_plot = True
def main(app,maxiter,n_repeats):
    rs = RandomSearch(app, maxiter, n_repeats)
    return rs.for_range_of_iter()

if __name__ == '__main__':
    app = Application(Sampler(Sim()))
    print(f"FINAL RESULTS RANDOMSEARCH: {main(app,(1, 20),1)}")
