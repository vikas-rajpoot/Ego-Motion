import plotly.graph_objects as go
import pandas as pd

# Load the pose data 
df = pd.read_csv("./data/global_pose.csv")  # Replace with actual file
# df_pred = pd.read_csv("predicted_pose.csv")  # Replace with predicted file

path = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_global/poses_T.txt"  

df_pred = pd.read_csv(path, header=None, sep=' ') 

df_pred.columns = ['tx', 'ty', 'tz', 'r11', 'r12', 'r13', 
                                        'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# Extract X, Y, Z coordinates
x_actual, y_actual, z_actual = df['tx'], df['ty'], df['tz']
x_pred, y_pred, z_pred = df_pred['tx'], df_pred['ty'], df_pred['tz']

# Create a 3D plot
fig = go.Figure()

# Actual trajectory (Ground Truth)
fig.add_trace(
    go.Scatter3d(
        x=x_actual, y=y_actual, z=z_actual,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=4, color='blue'),
        name="Actual Path"
    )
)

# Predicted trajectory
fig.add_trace(
    go.Scatter3d(
        x=x_pred, y=y_pred, z=z_pred,
        mode='lines+markers',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=4, color='red'),
        name="Predicted Path"
    )
)

# Set layout options
fig.update_layout(
    title="Actual vs Predicted Drone Trajectory",
    scene=dict(
        xaxis_title="X Position",
        yaxis_title="Y Position",
        zaxis_title="Z Position",
        aspectmode="cube"
    ),
    legend=dict(x=0, y=1)
)

# Show the plot
fig.show()


