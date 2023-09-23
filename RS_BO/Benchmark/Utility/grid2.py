import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# The original function to generate randomized grid
def generate_randomized_grid(lower_bound, upper_bound, num_iterations):
    bounds = upper_bound - lower_bound
    equidistant_points = np.linspace(lower_bound, upper_bound, num_iterations, endpoint=False)
    random_shifts = np.random.uniform(-bounds / (2 * num_iterations), bounds / (2 * num_iterations), num_iterations)
    randomized_points = equidistant_points + random_shifts
    randomized_points = randomized_points % upper_bound
    randomized_points.sort()
    return randomized_points


# Initialize dictionary to hold frequency counts
frequency_count = defaultdict(int)

# Define parameters
bounds = 90
iterations = 30
num_executions = 10000

# Execute the algorithm multiple times
for _ in range(num_executions):
    randomized_grid = generate_randomized_grid(0,90, iterations)
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
