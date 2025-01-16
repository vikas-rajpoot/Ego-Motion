import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load the pose data 
df_global_pose = pd.read_csv("./vikas_data/global_pose.csv")

# Extract translation (position) and rotation (orientation) data
x = df_global_pose['tx']
y = df_global_pose['ty']
z = df_global_pose['tz']

# Extract rotation matrix components
r11, r12, r13 = df_global_pose['r11'], df_global_pose['r12'], df_global_pose['r13']
r21, r22, r23 = df_global_pose['r21'], df_global_pose['r22'], df_global_pose['r23']
r31, r32, r33 = df_global_pose['r31'], df_global_pose['r32'], df_global_pose['r33']

# Calculate orientation vectors for visualization
arrow_length = 0.5  # Adjust the length of the orientation arrows
u = r11 * arrow_length
v = r21 * arrow_length
w = r31 * arrow_length

# Create 3D plot
fig = go.Figure()

# Add trajectory line
fig.add_trace(
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color=np.arange(len(x)), colorscale='Viridis', opacity=0.8),
        name='Trajectory',
        text=[f"Frame: {i}<br>Position: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})" for i in range(len(x))],
        hoverinfo='text'
    )
)

# Add orientation arrows
fig.add_trace(
    go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        sizemode='absolute',
        sizeref=arrow_length / 2,  # Adjust arrow scaling
        anchor='tail',
        colorscale='Blues',
        name='Orientation'
    )
)

# Customize layout
fig.update_layout(
    title="Drone/Camera Trajectory and Orientation",
    scene=dict(
        xaxis_title="X (Position)",
        yaxis_title="Y (Position)",
        zaxis_title="Z (Position)",
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Adjust camera angle
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.1, y=0.9)
)

# Show the plot
fig.show() 
fig.write_html("./vikas_data/drone_trajectory_orientation.html")
print("Interactive visualization saved as drone_trajectory_orientation.html")

