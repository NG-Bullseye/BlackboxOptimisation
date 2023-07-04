import numpy as np
import plotly.graph_objects as go
if __name__ == '__main__':

    # Create a grid of points
    x = np.linspace(-5, 5, 100)  # 100 points between -5 and 5
    y = np.linspace(-5, 5, 100)  # 100 points between -5 and 5
    x, y = np.meshgrid(x, y)

    # Apply the function to each point in the grid
    z =(np.sin(x)+np.sin(y)) / 2

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

    # Update layout for a better view
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=10
        ),
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        )
    )

    # Show the plot
    fig.show()
