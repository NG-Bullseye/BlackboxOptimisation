import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':

    # parameters for the perturbation function
    a = 1.0
    mu = 1.0  # center of the wave
    width = 0.1
    y_offset = 0.0

    # create a grid of (x, y) coordinates
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    x, y = np.meshgrid(x, y)

    # convert to polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # calculate the perturbation
    perturbation = a * ((r - mu) / width) * np.exp(-((r - mu) / width) ** 2)+y_offset

    # make half of the donut positive and half negative
    # add a smooth transition with a sigmoid function
    smooth_factor_theta = 10  # adjust this parameter to change the smoothness of the transition
    transition = 1 / (1 + np.exp(-smooth_factor_theta * np.sin(theta)))

    z = perturbation * (2 * transition - 1)  # scale and shift to range from -1 to 1

    # plot the Gaussian donut
    surface = go.Surface(x=x, y=y, z=z)
    fig = go.Figure(data=[surface])

    fig.update_layout(
        title="3D Gaussian Donut",
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()
