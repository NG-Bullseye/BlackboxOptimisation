import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gradient_vector(x):
    x1 = [-np.sin(xi[0]) / 5 for xi in x]
    x2 = [-np.sin(xi[1]) / 5 for xi in x]
    gradient = np.array([x1, x2])
    norm = np.linalg.norm(gradient)
    unit_gradient = gradient / norm
    return unit_gradient

def perturbation(x, mu, width, a, y_offset):
    perturbation = a * np.exp(-np.sum(((x - mu) / width) ** 2, axis=-1)) - \
                   a * np.exp(-np.sum(((x + mu) / width) ** 2, axis=-1)) + y_offset
    return perturbation.flatten()
if __name__ == '__main__':

    # Example usage
    x1_test = np.linspace(0, 10, 100)
    x2_test = np.linspace(0, 10, 100)
    mu = np.array([5, 5])
    PERT_WIDTH = 1
    PERT_SCALE = 1

    a = PERT_SCALE * gradient_vector(mu.reshape(1, -1))
    y_offset = 0

    X, Y = np.meshgrid(x1_test, x2_test)
    Z = perturbation(np.dstack([X, Y]).reshape(-1, 2), mu, PERT_WIDTH, a, y_offset).reshape(X.shape)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='viridis'))

    fig.update_layout(
        title=f'Perturbation at (5,5) with WIDTH={PERT_WIDTH}, SCALE={PERT_SCALE}',
        scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis')
    )

    fig.show()
