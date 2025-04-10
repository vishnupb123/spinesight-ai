import plotly.graph_objects as go
import numpy as np

def generate_3d_spine_plot(input_data):
    # Extract features from input_data
    pelvic_incidence = input_data.get('pelvic_incidence', 50)  # Default if missing
    pelvic_tilt = input_data.get('pelvic_tilt', 10)
    lumbar_lordosis_angle = input_data.get('lumbar_lordosis_angle', 20)
    sacral_slope = input_data.get('sacral_slope', 30)
    pelvic_radius = input_data.get('pelvic_radius', 200)
    degree_spondylolisthesis = input_data.get('degree_spondylolisthesis', 10)

    # Simulate 3D spine model based on input features
    # Example spine model (just a simple representation for now)
    x = np.linspace(-5, 5, 100)  # X-axis for the spine length
    y = pelvic_tilt * np.sin(x)  # Curvature based on pelvic tilt
    z = lumbar_lordosis_angle * np.cos(x)  # Z-axis for spine depth

    # Creating the 3D plot
    fig = go.Figure()

    # Plot the spine curve based on input features
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='blue', width=5),
        name='Spine'
    ))

    # Add the pelvis as a reference (e.g., a sphere or a flat surface)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[pelvic_incidence], z=[0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Pelvic Incidence'
    ))

    # Add a representation for spondylolisthesis (slippage of spine)
    slip_x = [0, degree_spondylolisthesis]  # Example slip representation
    slip_y = [0, degree_spondylolisthesis]
    slip_z = [0, 0]
    
    fig.add_trace(go.Scatter3d(
        x=slip_x, y=slip_y, z=slip_z,
        mode='lines+markers',
        line=dict(color='orange', width=6),
        marker=dict(size=8),
        name='Spondylolisthesis'
    ))

    # Customize plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Spine Length',
            yaxis_title='Curvature',
            zaxis_title='Depth',
        ),
        title="3D Spine Visualization",
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # Convert to HTML to embed in Flask
    plotly_3d_div = fig.to_html(full_html=False)
    
    return plotly_3d_div
