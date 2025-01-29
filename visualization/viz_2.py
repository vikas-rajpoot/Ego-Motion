import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

# Sample DataFrame (Replace this with your actual df_global_pose)
df_global_pose = pd.read_csv("./data/pose.csv") 

# Create a 3D figure
fig = go.Figure()

# Plot drone positions
fig.add_trace(go.Scatter3d(
    x=df_global_pose['tx'], y=df_global_pose['ty'], z=df_global_pose['tz'],
    mode='markers', marker=dict(size=5, color='blue'), name='Drone Position'
))

# Plot orientation arrows
arrow_length = 0.5  # Length of orientation vectors
for _, row in df_global_pose.iterrows():
    position = np.array([row['tx'], row['ty'], row['tz']])
    rotation = R.from_euler('xyz', [row['rx'], row['ry'], row['rz']], degrees=True)
    direction = rotation.apply([1, 0, 0])  # Forward direction
    end_point = position + arrow_length * direction
    
    fig.add_trace(go.Scatter3d(
        x=[position[0], end_point[0]],
        y=[position[1], end_point[1]],
        z=[position[2], end_point[2]],
        mode='lines',
        line=dict(color='red', width=5),
        name='Orientation'
    ))

# Configure layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    title='Drone Position and Orientation in 3D'
)

fig.show()
