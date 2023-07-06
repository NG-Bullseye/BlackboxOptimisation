import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gradient_vector(x):
    x1 = -np.sin(x[0][0]) / 5
    x2 = -np.sin(x[0][1]) / 5
    gradient = np.array([x1, x2])
    return gradient

def f(x):
    return (np.sin(x[0]) + np.sin(x[1])) / 5+0.5
def perturbation(x, mu, width, a, y_offset):
    perturbation = a * np.exp(-np.sum(((x - mu) / width) ** 2, axis=-1)) - \
                   a * np.exp(-np.sum(((x + mu) / width) ** 2, axis=-1)) + y_offset
    return perturbation

if __name__ == '__main__':

    # Example usage
    x1_test = np.linspace(0, 10, 100)
    x2_test = np.linspace(0, 10, 100)
    mu = np.array([5.0, 6.0])  # change this line
    f_mu=f(mu)
    PERT_WIDTH = 1
    PERT_SCALE = 1

    # Shift mu by the gradient
    mu += PERT_SCALE * gradient_vector(mu.reshape(1, -1))
    y_offset = 0

    X, Y = np.meshgrid(x1_test, x2_test)
    Z = perturbation(np.dstack([X, Y]).reshape(-1, 2), mu, PERT_WIDTH, PERT_SCALE, y_offset)
    Z = Z.reshape(X.shape)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='viridis'))
    fig.add_trace(
        go.Scatter3d(x=[mu[0]], y=[mu[1]], z=[f_mu],
                     mode='markers', marker=dict(size=4, color='red')),
        row=1, col=1)

    fig.update_layout(
        title=f'Perturbation at (5,5) with WIDTH={PERT_WIDTH}, SCALE={PERT_SCALE}',
        scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis')
    )

    fig.show()