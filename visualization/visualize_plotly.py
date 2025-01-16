import plotly.graph_objects as go
import pandas as pd

# Load global_pose data from the CSV
df_global_pose = pd.read_csv("./vikas_data/global_pose.csv")

# Extract translations
x = df_global_pose['tx']
y = df_global_pose['ty']
z = df_global_pose['tz']

# Create a 3D scatter plot for the trajectory
fig = go.Figure()

# Add trajectory as a 3D line
fig.add_trace(
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color='red', opacity=0.8),
        name='Camera Trajectory',
        hoverinfo='text',
        text=[f"Frame: {i}" for i in range(len(x))]
    )
)

# Set axis labels and plot title
fig.update_layout(
    title="Camera Trajectory in 3D Space",
    scene=dict(
        xaxis_title="X (Translation)",
        yaxis_title="Y (Translation)",
        zaxis_title="Z (Translation)"
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

# Show the plot
fig.show()

fig.write_html("./vikas_data/camera_trajectory.html") 
print("Plot saved as camera_trajectory.html")

