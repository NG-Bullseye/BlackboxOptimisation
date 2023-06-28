import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import gpflow
from gpflow.utilities import to_default_float


class ConstantMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, c):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.c = gpflow.Parameter(c, dtype=gpflow.default_float())

    def __call__(self, X):
        return tf.fill(tf.concat([tf.shape(X)[:-1], [1]], 0), self.c)

np.random.seed(43)
# Assuming we have some data
X = np.random.rand(10, 1)
y = np.sin(X).reshape(-1, 1)  # Use reshape instead of ravel

# Define the kernel
kernel = gpflow.kernels.RBF(lengthscales=0.5) + gpflow.kernels.White(variance=1.0)

# Create the mean function
meanoffset=6
meanf = ConstantMean(meanoffset)  # This will shift the predictions vertically by 2.5 units

# Create and train the Gaussian Process model
m = gpflow.models.GPR(data=(X, y), kernel=kernel, mean_function=meanf)

# Optimize the model parameters
opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, m.trainable_variables)

# Predict at new points
X_new = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred, var_pred = m.predict_y(X_new)

# Plot the results
plt.figure()
plt.plot(X_new, y_pred, label='GP mean')
plt.fill_between(X_new[:, 0], y_pred[:, 0] - np.sqrt(var_pred)[:, 0], y_pred[:, 0] + np.sqrt(var_pred)[:, 0], alpha=0.2, label='95% Confidence Interval')
plt.scatter(X, y, color='red', label='Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Process Regression\nMean Offset: {:.2f}'.format(meanoffset))
plt.show()
