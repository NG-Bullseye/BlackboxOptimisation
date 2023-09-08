import random
import time
import matplotlib.pyplot as plt
from Application import Application, Sampler  # Make sure to import Application and Sampler from your file
from RS_BO.Utility.Sim import Sim

# Initialize the application with a Sampler instance
class Gridsearch:
    def __init__(self,app,n_repeats,maxiter):
        # Example usage
        self.n_repeats=n_repeats
        self.maxiter=maxiter
        self.app=app
        self.early_stop_threshold = 95
        self.enable_plot = True
        self.enable_plot_grid = True
        # grid_search([0, 90], num_iterations=5),

    def plot_grid(self,grid_points, bounds):
        fig, ax = plt.subplots()
        ax.scatter(grid_points, [0] * len(grid_points), marker='x', color='red', label='Grid Points')

        # Plot yaw_acc points
        sortedDict = dict(sorted(self.app.sampler.shift_dict_to_positive(self.app.sampler.yaw_acc).items(), key=lambda item: item[0]))
        x_acc_points = list(sortedDict.keys())
        y_acc_points = list(sortedDict.values())
        ax.scatter(x_acc_points, y_acc_points, marker='o', color='blue', label='Objective Function Values')

        ax.set_xlim(bounds)
        ax.set_title('Grid Points and Objective Function Over Search Space')
        plt.xlabel('Search Space')
        plt.ylabel('Function Value (Objective)')
        ax.legend()
        plt.show()
    def plot_data(self,x_values, y_values, enable_plot):
        if enable_plot:
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Number of Iterations')
            ax1.set_ylabel('Average Percentage Close to Global Maximum', color=color)
            ax1.plot(x_values, y_values, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            plt.title('Grid Search Performance')
            plt.grid(True)
            plt.savefig('grid_search_performance.png')
            plt.show()

    def grid_search(self,bounds, num_iterations):
        lower_bound, upper_bound = bounds
        step_size = (upper_bound - lower_bound) / (num_iterations - 1)
        max_found_y = float('-inf')
        grid_points = []  # Added list to collect individual grid points

        for i in range(num_iterations):
            x_value = lower_bound + i * step_size
            y_value = self.app.sampler.f_discrete_real_data_x(x_value)
            max_found_y = max(max_found_y, y_value)
            print(f"y_value:{y_value} x_value:{x_value}")
            self.app.sampler.calculateRegret([y_value])
            grid_points.append(x_value)  # Collect individual grid points
        global_max = self.app.sampler.getGlobalOptimum_Y()
        percentage_close = (max_found_y / global_max) * 100
        return percentage_close, grid_points  # Return individual grid points

    def for_range_of_iter(self, maxiter, early_stop_threshold=0.1, num_runs=10, enable_plot=False, enable_plot_grid=False):
        start, end = maxiter
        iter_values = []

        for num_iterations in range(start, end + 1):
            avg_percentage_close = 0

            for run in range(num_runs):
                current_time = time.time()
                random.seed(int(current_time * 1e9))

                percentage_close, grid_points = self.grid_search([0, 90], num_iterations)
                avg_percentage_close += percentage_close
                if enable_plot_grid:
                    self.plot_grid(grid_points, [0, 90])  # Plot the grid points of a single run

            avg_percentage_close /= num_runs

            print(f"For num_iterations = {num_iterations}, Average Percentage Close: {avg_percentage_close}%")

            if avg_percentage_close >= early_stop_threshold:
                print("Stopping early as the average percentage close reached the given threshold.")
                break

            iter_values.append((num_iterations, avg_percentage_close))

        if enable_plot:
            self.plot_data([i[0] for i in iter_values], [i[1] for i in iter_values],True)
        return iter_values



def main(app,maxiter,n_repeats):

    gridsearch=Gridsearch(app,maxiter,n_repeats)
    return gridsearch.for_range_of_iter(maxiter, num_runs=1)
if __name__ == '__main__':

    print(f"FINAL RESULTS GRIDSEARCH: {main(app,maxiter = (2, 10),n_repeats=1)}")