# # import pandas as pd

# # # Example data
# # df = pd.read_csv("./data/global_pose.csv")  

# # # Apply cumulative sum transformation
# # df['values'] = df['values'].cumsum()

# # print(df) 

# import plotly.graph_objects as go
# import pandas as pd
 
# # Load the pose data (Assuming the file contains 'tx', 'ty', 'tz' for x, y, z coordinates) 
# path = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_global/poses_T.txt"  


# df = pd.read_csv(path, header=None, sep=' ')   

# df.columns = ['tx', 'ty', 'tz', 'r11', 'r12', 'r13', 
#                                         'r21', 'r22', 'r23', 'r31', 'r32', 'r33'] 

# # Extract X, Y, Z coordinates
# x = df['tx']
# y = df['ty']
# z = df['tz']

# # Create a 3D trajectory plot
# fig = go.Figure()

# # Add the drone's trajectory
# fig.add_trace(
#     go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='lines+markers',
#         line=dict(color='blue', width=2),
#         marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8),
#         name="Drone Path"
#     )
# )

# # Add layout styling
# fig.update_layout(
#     title="Drone 3D Position Visualization",
#     scene=dict(
#         xaxis_title="X Position",
#         yaxis_title="Y Position",
#         zaxis_title="Z Position",
#         aspectratio=dict(x=1, y=1, z=1),
#         camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Adjust camera angle
#     ),
#     margin=dict(l=0, r=0, b=0, t=40)
# )

# # Show the plot
# fig.show()




# import pandas as pd

# # Example data
# df = pd.read_csv("./data/global_pose.csv")  

# # Apply cumulative sum transformation
# df['values'] = df['values'].cumsum()

# print(df) 

import plotly.graph_objects as go
import pandas as pd
 
# Load the pose data 
df = pd.read_csv("./data/global_pose.csv") 


print("df global pose : ",df.columns)   

df['tx'] = df['tx'].cumsum() 
df['ty'] = df['ty'].cumsum() 
df['tz'] = df['tz'].cumsum() 

# # Extract X, Y, Z coordinates
x = df['tx']
y = df['ty']
z = df['tz']

# Create a 3D trajectory plot
fig = go.Figure()

# Add the drone's trajectory
fig.add_trace(
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8),
        name="Drone Path"
    )
)

# Add layout styling
fig.update_layout(
    title="Drone 3D Position Visualization",
    scene=dict(
        xaxis_title="X Position",
        yaxis_title="Y Position",
        zaxis_title="Z Position",
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Adjust camera angle
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the plot
fig.show()







