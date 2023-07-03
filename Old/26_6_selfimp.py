import numpy as np
import matplotlib.pyplot as plt

# Define points and function values
x = np.array([1, 2])
f_values = np.array([0, 1])
df_values = np.array([2, 0])

# Define kernel function and its derivative
def kernel(x, y):
    return np.exp(-0.5 * (x - y) ** 2)

def kernel_derivative(x, y):
    return (y - x) * kernel(x, y)

def kernel_second_derivative(x, y):
    return (1 - (y - x) ** 2) * kernel(x, y)

# Construct covariance matrix
K = np.zeros((4, 4))
for i in range(2):
    for j in range(2):
        K[i, j] = kernel(x[i], x[j])
        K[i, j + 2] = kernel_derivative(x[i], x[j])
        K[i + 2, j] = kernel_derivative(x[j], x[i])  # Derivatives are not symmetric
        K[i + 2, j + 2] = kernel_second_derivative(x[i], x[j])

x_stars = np.linspace(0, 4, 1000)  # range of x_star values
mu_stars = []
upper_bounds = []
lower_bounds = []

for x_star in x_stars:
    # Construct covariance vector K_*
    K_star = np.zeros(4)
    for i in range(2):
        K_star[i] = kernel(x_star, x[i])
        K_star[i + 2] = kernel_derivative(x_star, x[i])

    # Calculate predictive mean mu* and variance var*
    f_df_values = np.concatenate([f_values, df_values])
    K_inv_f = np.linalg.solve(K, f_df_values)
    mu_star = K_star @ K_inv_f
    var_star = kernel(x_star, x_star) - K_star @ np.linalg.solve(K, K_star)

    # Apply sigmoid transformation to mu_star and var_star
    mu_star_transformed = 1 / (1 + np.exp(-mu_star))
    var_star_transformed = (1 / (1 + np.exp(-mu_star))) ** 2 * var_star

    # Compute the standard deviation and confidence bounds
    std_transformed = np.sqrt(var_star_transformed)
    upper_bound = mu_star_transformed + 2 * std_transformed
    lower_bound = mu_star_transformed - 2 * std_transformed

    mu_stars.append(mu_star_transformed)
    upper_bounds.append(upper_bound)
    lower_bounds.append(lower_bound)

# Define original function for plotting
f_x = -(x_stars - 2) ** 2 + 1

# Plotting
plt.plot(x, f_values, 'bo', label='Data points')
plt.plot(x_stars, mu_stars, 'r-', label='Mean predictions')
plt.plot(x_stars, f_x, 'g-', label='Original function')
plt.fill_between(x_stars, lower_bounds, upper_bounds, color='r', alpha=0.1, label='Confidence band')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Confidence interval of prediction at x*')
plt.legend()
plt.show()
