import numpy as np
import matplotlib.pyplot as plt


# Objective function: f(x) = x^2 (to be minimized)
def objective_function(x):
    return x ** 2


# Initialize variables
n_iterations = 10
best_scores = []
best_score = float('inf')

# Random Search
for i in range(n_iterations):
    candidate = np.random.uniform(-10, 10)
    score = objective_function(candidate)

    if score < best_score:
        best_score = score

    best_scores.append(best_score)

# Plotting the learning curve
plt.plot(best_scores)
plt.xlabel('Iterations')
plt.ylabel('Best Score')
plt.title('Random Search Learning Curve')
plt.show()
