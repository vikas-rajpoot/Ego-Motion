import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
# Load global_pose data from the CSV
df_global_pose = pd.read_csv("./vikas_data/global_pose.csv")

# Extract translations
x = df_global_pose['tx']
y = df_global_pose['ty']
z = df_global_pose['tz']

# Plot 3D trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Camera Trajectory', marker='o')
ax.set_xlabel('X (Translation)')
ax.set_ylabel('Y (Translation)')
ax.set_zlabel('Z (Translation)')
ax.set_title("Camera Trajectory in 3D Space")
ax.legend()
plt.show()


