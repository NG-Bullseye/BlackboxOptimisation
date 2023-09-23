import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim


# The original function to generate randomized grid
class test:
    def __init__(self):
        self.app = Application(Sampler(Sim()))
        # Initialize dictionary to hold frequency counts
        frequency_count = defaultdict(int)
        bounds = 90
        iterations = 30
        num_executions = 10000
        # Execute the algorithm multiple times

        for _ in range(num_executions):
            randomized_grid = self.random_search(90, iterations)
            for point in randomized_grid:
                # Round to a certain decimal to group the points
                rounded_point = round(point, 2)
                frequency_count[rounded_point] += 1
        # Calculate probabilities and prepare data for plotting
        keys = sorted(frequency_count.keys())
        values = [frequency_count[k] / num_executions for k in keys]

        # Plotting
        plt.bar(keys, values, width=0.1)
        plt.xlabel('Generated Points')
        plt.ylabel('Probability')
        plt.title('Probability Distribution of Generated Points')
        plt.show()

    def random_search(self, bounds, iter):
        lower_bound, upper_bound = bounds
        max_found_y = float('-inf')
        cum_reg = 0
        for i in range(iter):
            # Single random point
            x_value = np.random.uniform(lower_bound, upper_bound)
            y_value = self.app.sampler.f_discrete_real_data(np.array([x_value]))
            y_value = y_value[0]
            max_found_y = max(max_found_y, y_value)
            cum_reg += abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())
            # print(f"iter{iter} y_value{y_value} max_found_y{max_found_y}  i{i} cum_reg{cum_reg} reg{abs(max_found_y - self.app.sampler.getGlobalOptimum_Y())}")

            # print(f"Single iteration max_found_y: {max_found_y}")

            # global_max = self.app.sampler.getGlobalOptimum_Y()
            # percentage_close = (max_found_y / global_max) * 100

        return max_found_y, cum_reg  # Just return max_found_y here




