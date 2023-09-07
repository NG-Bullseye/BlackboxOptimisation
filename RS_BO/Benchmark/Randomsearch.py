import numpy as np
import random
import time
import matplotlib.pyplot as plt
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
def random_search(bounds, num_points, app):
    lower_bound, upper_bound = bounds
    max_found_y = float('-inf')
    random_points = np.random.uniform(lower_bound, upper_bound, num_points)

    for i, x_value in enumerate(random_points):
        # Convert x_value to a 1D array before passing it to f_discrete_real_data
        y_value = app.sampler.f_discrete_real_data(np.array([x_value]))
        y_value = y_value[0]  # Assuming y_value is a list with one element
        max_found_y = max(max_found_y, y_value)
        print(f"Iterer: {i} max_found_y: {max_found_y}")

    global_max = app.sampler.getGlobalOptimum_Y()
    percentage_close = (max_found_y / global_max) * 100
    return percentage_close

def plot_data(x_values, y_values, enable_plot):
    if enable_plot:
        plt.figure()
        plt.plot(x_values, y_values, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Percentage Close to Global Maximum')
        plt.title('Random Search Performance')
        plt.grid(True)
        plt.savefig('random_search_performance.png')
        plt.show()

def for_range_of_iter(iter_interval, early_stop_threshold, num_runs=10, enable_plot=False):
    start, end = iter_interval
    iter_values = []

    for num_points in range(start, end + 1):
        avg_percentage_close = 0

        for run in range(num_runs):
            current_time = time.time()
            random.seed(int(current_time * 1e9))
            app = Application(Sampler(0.1, Sim()))

            percentage_close = random_search([0, 90], num_points, app)
            avg_percentage_close += percentage_close

        avg_percentage_close /= num_runs

        print(f"For num_points = {num_points}, Average Percentage Close: {avg_percentage_close}%")

        iter_values.append((num_points, avg_percentage_close))  # Append before checking for early stopping

        if avg_percentage_close >= early_stop_threshold:
            print("Stopping early as the average percentage close reached the given threshold.")
            break  # Break after appending

    if enable_plot:
        plot_data([i[0] for i in iter_values], [i[1] for i in iter_values], True)


# Example usage
iter_interval = (1, 20)
early_stop_threshold = 95
enable_plot = True
for_range_of_iter(iter_interval, early_stop_threshold, num_runs=20, enable_plot=enable_plot)
