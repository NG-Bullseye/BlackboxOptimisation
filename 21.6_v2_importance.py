import numpy as np
import GPy
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)+1

def df(x):
    return np.cos(x)

if __name__ == '__main__':
    # 1. Define observed inputs
    x = np.array([[1], [2]])
    print("Defined observed inputs at x = 1 and x = 2. x:", x)

    # 2. Define observed function values
    y = f(x)
    print("Calculated observed function values at the observed inputs. y:", y)

    # 3. Define observed derivative values
    dy = np.array(df(x))
    print("Calculated derivative of the function at the observed inputs. dy:", dy)

    # Introduce the hyperparameters for controlling the influence of the derivative information
    derivative_scale = 3  # Change this to any value to control the influence of the derivative information
    derivative_length = 0.5  # Change this to any value to control the "reach" of the derivative information
    dy_scaled = dy * derivative_scale
    print("Scaled the derivative values by the hyperparameter. dy_scaled:", dy_scaled)

    # 4. Stack x, y and dy_scaled, while extending x with the derivative length for dy_scaled
    X = np.vstack([x, x + derivative_length])  # Stack x on itself for function and derivative values
    Y = np.vstack([y, dy_scaled])  # Stack y and dy_scaled
    print("Stacked inputs and outputs for both function values and derivative values. X:", X, "Y:", Y)

    # 5. Define which are values and which are derivatives
    D = np.vstack([np.zeros_like(x), np.ones_like(x)])
    print("Defined an indicator matrix D to specify which outputs are function values and which are derivatives. D:", D)

    # 6. Specify input_dim (1) and variance (1) for the RBF kernel
    k = GPy.kern.RBF(2, variance=1)  # Changed input_dim from 1 to 2
    print("Defined the kernel to be used in the Gaussian Process.")

    # 7. Concatenate X and D to form the new input
    X_new = np.hstack((X, D))
    print("Concatenated the inputs and the D matrix to form the new inputs to the Gaussian Process. X_new:", X_new)

    # 8. Build GP model
    model = GPy.models.GPRegression(X_new, Y, k)
    print("Built the Gaussian Process model with the specified inputs, outputs, and kernel.")

    # 9. Optimize the model
    model.optimize()
    print("Optimized the parameters of the Gaussian Process model.")

    # Define a new x space for prediction
    x_new_space = np.linspace(0, 3, 100).reshape(-1, 1)
    print("Defined a new x space for making predictions with the GP model. x_new_space:", x_new_space)

    # Prepare this space for model prediction
    x_star = np.hstack([x_new_space, np.zeros_like(x_new_space)])
    print("Prepared the new x space for making predictions with the GP model. x_star:", x_star)

    # 10. Predict at new points
    mu_star, var_star = model.predict(x_star)
    print("Made predictions at the new x points. mu_star:", mu_star, "var_star:", var_star)

    # 11. Compute the 95% confidence interval
    CI = [mu_star - 2 * np.sqrt(var_star), mu_star + 2 * np.sqrt(var_star)]
    print(f"Computed the 95% confidence interval for the predictions. CI: {CI}")

    # 12. Plotting
    x_for_plot = np.linspace(0, 2 * np.pi, 100)
    y_for_plot = f(x_for_plot)
    print("Prepared the x values and function values for plotting.")

    plt.figure(figsize=(10, 5))
    plt.plot(x_for_plot, y_for_plot, label='f(x)')

    # Plot the confidence interval
    plt.fill_between(x_new_space.ravel(), CI[0].ravel(), CI[1].ravel(), color='gray', alpha=0.5, label='Confidence Interval')
    print("Plotted the confidence interval for the predictions.")

    # Plot the GP mean
    plt.plot(x_new_space, mu_star, '--', color='black', label='GP mean')
    print("Plotted the mean of the Gaussian Process predictions.")

    # Plot the sampled points
    plt.scatter(x, y, color='r', label='Sampled points')
    print("Plotted the sampled points and their corresponding function values.")

    # Draw the arrow for derivative
    for i in range(x.shape[0]):
        tangent_direction = np.array([1, dy[i, 0]]) / np.linalg.norm([1, dy[i, 0]])
        arrow_end = [x[i, 0] + 0.2 * tangent_direction[0],
                     f(x[i, 0]) + 0.2 * tangent_direction[1]]
        plt.arrow(x[i, 0], f(x[i, 0]), 0.2 * tangent_direction[0], 0.2 * tangent_direction[1],
                  facecolor='green', edgecolor='green', head_width=0.05, head_length=0.1)
        print(f"Drawn tangent arrow at x = {x[i, 0]} to represent derivative.")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gaussian Process Regression and Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Finished plotting.")
