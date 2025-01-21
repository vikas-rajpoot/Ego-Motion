import matplotlib.pyplot as plt
import numpy as np

# Create some sample data (a simple gradient)
data = np.random.rand(100, 100)

# Create the figure and axis
fig, ax = plt.subplots()

# Display the image
ax.imshow(data, cmap='viridis')

# Remove the axes
ax.axis('off')

# Adjust the layout to remove borders/padding
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save the image without borders or padding
plt.savefig('data/output_image.png', bbox_inches='tight', pad_inches=0)

# Close the figure to free memory
plt.close(fig)


