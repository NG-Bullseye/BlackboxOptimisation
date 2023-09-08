import random
import time
import matplotlib.pyplot as plt
from Application import Application, Sampler  # Make sure to import Application and Sampler from your file
from RS_BO.Utility.Sim import Sim

# Initialize the application with a Sampler instance
class Gridsearch:
    def __init__(self,app,maxiter,n_repeats):
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
        if num_iterations == 1:
            step_size = 0  # or whatever makes sense in this context
        else:
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

    def for_range_of_iter(self, enable_plot=False, enable_plot_grid=False):
        iter_values = []

        for num_iterations in range(1,self.maxiter+1):
            avg_percentage_close = 0

            for run in range(self.n_repeats):
                current_time = time.time()
                random.seed(int(current_time * 1e9))

                percentage_close, grid_points = self.grid_search([0, 90], num_iterations)
                avg_percentage_close += percentage_close
                if enable_plot_grid:
                    self.plot_grid(grid_points, [0, 90])  # Plot the grid points of a single run

            avg_percentage_close /= self.n_repeats

            print(f"For num_iterations = {num_iterations}, Average Percentage Close: {avg_percentage_close}%")

            if avg_percentage_close >= self.early_stop_threshold:
                print("Stopping early as the average percentage close reached the given threshold.")
                break

            iter_values.append((num_iterations, avg_percentage_close))

        if enable_plot:
            self.plot_data([i[0] for i in iter_values], [i[1] for i in iter_values],True)
        return iter_values

    def calculate_averages(self,iter_values, n_repeats):
        average_optimal_fxs = [y / n_repeats for (_, y) in iter_values]
        global_max = iter_values[-1][1]  # Assuming the last element holds the global max
        average_cum_reg = [abs(y - global_max) / n_repeats for (_, y) in iter_values]

        return average_optimal_fxs, average_cum_reg


# Example usage




def main(app,maxiter,n_repeats):
    gridsearch=Gridsearch(app,maxiter+1,n_repeats)#+1 because there is no initial gussee like in bo
    iter_values = gridsearch.for_range_of_iter()
    return  gridsearch.calculate_averages(iter_values, n_repeats)
if __name__ == '__main__':
    app = Application(Sampler(Sim()))
    print(f"FINAL RESULTS GRIDSEARCH: {main(app,1,1)}")