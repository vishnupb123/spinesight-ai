import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to generate the heatmap visualization
import plotly.graph_objs as go



def generate_comparison_bar_chart(input_data):
    categories = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                  'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

    healthy_ranges = {
        'pelvic_incidence': (45, 65),
        'pelvic_tilt': (10, 20),
        'lumbar_lordosis_angle': (35, 60),
        'sacral_slope': (35, 50),
        'pelvic_radius': (11, 129),
        'degree_spondylolisthesis': (0, 5)
    }

    measured = [input_data[feature] for feature in categories]
    healthy_min = [healthy_ranges[feature][0] for feature in categories]
    healthy_max = [healthy_ranges[feature][1] for feature in categories]

    fig = go.Figure()

    # Healthy Range Bars
    fig.add_trace(go.Bar(
        x=categories,
        y=[max_val - min_val for min_val, max_val in zip(healthy_min, healthy_max)],
        base=healthy_min,
        name='Healthy Range',
        marker_color='lightgreen',
        opacity=0.6
    ))

    # Measured Value Markers
    fig.add_trace(go.Scatter(
        x=categories,
        y=measured,
        mode='markers+text',
        name='Patient Value',
        marker=dict(size=12, color='crimson'),
        text=[f"{val:.1f}" for val in measured],
        textposition="top center"
    ))

    fig.update_layout(
        title="Spinal Metrics: Patient vs Healthy Ranges",
        yaxis_title="Value",
        barmode='overlay',
        height=400,
        width=600,
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white'
    )
    return fig.to_html(full_html=False)


# Function to generate the radar chart visualization
import plotly.graph_objects as go

def generate_radar_chart(input_data):
    categories = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                  'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

    # Define medically realistic healthy ranges
    healthy_ranges = {
        'pelvic_incidence': (45, 65),
        'pelvic_tilt': (10, 20),
        'lumbar_lordosis_angle': (35, 60),
        'sacral_slope': (35, 50),
        'pelvic_radius': (11, 129),
        'degree_spondylolisthesis': (0, 5)
    }

    measured_values = [input_data[feature] for feature in categories]
    healthy_min = [healthy_ranges[feature][0] for feature in categories]
    healthy_max = [healthy_ranges[feature][1] for feature in categories]

    # Close the polygons
    categories += [categories[0]]
    measured_values += [measured_values[0]]
    healthy_min += [healthy_min[0]]
    healthy_max += [healthy_max[0]]

    radar_chart = go.Figure()

    # Add healthy range as filled band
    radar_chart.add_trace(go.Scatterpolar(
        r=healthy_max,
        theta=categories,
        fill=None,
        mode='lines',
        line=dict(color='lightgreen'),
        name='Healthy Max'
    ))

    radar_chart.add_trace(go.Scatterpolar(
        r=healthy_min,
        theta=categories,
        fill='tonext',
        mode='lines',
        line=dict(color='lightgreen'),
        name='Healthy Min'
    ))

    # Add measured values
    radar_chart.add_trace(go.Scatterpolar(
        r=measured_values,
        theta=categories,
        fill='toself',
        name='Measured Spine',
        line=dict(color='crimson'),
        opacity=0.7
    ))

    radar_chart.update_layout(
        title="Spinal Metrics vs Healthy Ranges",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(measured_values), max(healthy_max)) + 10],
                tickfont=dict(size=8)
            )
        ),
        showlegend=True,
        margin=dict(l=75, r=35, t=60, b=40),
        
    )

    return radar_chart.to_html(full_html=False)

