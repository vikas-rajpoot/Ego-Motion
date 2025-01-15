#   git config --global user.email "you@example.com"
#   git config --global user.name "Your Name" 

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "drone_data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Check the data
print(data.head())

# Plot the translational data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data['tx'], label='Translation X')
plt.plot(data['ty'], label='Translation Y')
plt.plot(data['tz'], label='Translation Z')
plt.title('Drone Translational Movements')
plt.xlabel('Time (or Frame Index)')
plt.ylabel('Translation (units)')
plt.legend()
plt.grid()

# Plot the rotational data
plt.subplot(2, 1, 2)
plt.plot(data['rx'], label='Rotation X (Roll)', color='r')
plt.plot(data['ry'], label='Rotation Y (Pitch)', color='g')
plt.plot(data['rz'], label='Rotation Z (Yaw)', color='b')
plt.title('Drone Rotational Movements')
plt.xlabel('Time (or Frame Index)')
plt.ylabel('Rotation (degrees or radians)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
