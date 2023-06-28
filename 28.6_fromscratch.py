import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPyOpt.methods import BayesianOptimization

# Define the polynomial function to optimize
def polynomial_function(x):
    return np.polyval([1, -2, 1], x)  # Example polynomial: x^2 - 2x + 1

# Bounds for the input variable x
bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, 1)}]

# Optimization objective: maximize the polynomial function
def objective(x):
    return -polynomial_function(x)

# Assume we have some input data
X = np.random.uniform(0, 1, (10, 1))  # 10 random inputs in the interval [0, 1]
Y = polynomial_function(X) + np.random.normal(0, 0.1, (10, 1))  # Generate noisy function values

# We create a GP model for regression
kernel = GPy.kern.RBF(input_dim=1)
model = GPy.models.GPRegression(X, Y, kernel)

# Optimize the model parameters
model.optimize()

# Bayesian Optimization
optimizer = BayesianOptimization(f=objective, domain=bounds, acquisition_type='EI')
optimizer.run_optimization(max_iter=10)

# Best found input value
best_x = optimizer.X[np.argmin(optimizer.Y)]

# Generate points for plotting the polynomial function
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
Y_plot, _ = model.predict(X_plot)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='black', label='Data')

# Plot the polynomial function
plt.plot(X_plot, Y_plot, color='blue', label='Polynomial')

# Plot the best found input value
plt.scatter(best_x, polynomial_function(best_x), color='red', marker='x', label='Best')

plt.legend()
plt.show()
