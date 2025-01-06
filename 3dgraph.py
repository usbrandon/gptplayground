import numpy as np
import matplotlib.pyplot as plt
# For 3D plotting:
from mpl_toolkits.mplot3d import Axes3D  # This import is needed in older versions of matplotlib

# Create some sample data:
# Blue points
blue_points = np.array([
    [1, 1, 1],
    [0.8, 0.5, 1.2]
])
# Orange points
orange_points = np.array([
    [1.0, 0.6, 0.8],
    [0.5, 1.0, 1.3],
    [0.2, 0.9, 0.5]
])

# Create a figure and 3D axes
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Option 1: Plot the coordinate axes (as 3 quivers):
# (you can also do simple lines if you prefer)
axis_length = 1.5
ax.quiver(0, 0, 0, axis_length, 0, 0, arrow_length_ratio=0.05, color='k')
ax.quiver(0, 0, 0, 0, axis_length, 0, arrow_length_ratio=0.05, color='k')
ax.quiver(0, 0, 0, 0, 0, axis_length, arrow_length_ratio=0.05, color='k')

# Option 2: Scatter the points
ax.scatter(blue_points[:,0], blue_points[:,1], blue_points[:,2], color='blue', s=50)
ax.scatter(orange_points[:,0], orange_points[:,1], orange_points[:,2], color='orange', s=50)

# Option 3: Draw dashed lines (vectors) from origin to each point
for (x, y, z) in blue_points:
    ax.plot([0, x], [0, y], [0, z], linestyle='--', color='gray')
for (x, y, z) in orange_points:
    ax.plot([0, x], [0, y], [0, z], linestyle='--', color='gray')

# Make the axes look a bit nicer
ax.set_xlim([0, axis_length])
ax.set_ylim([0, axis_length])
ax.set_zlim([0, axis_length])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust the view angle if desired
ax.view_init(elev=20, azim=30)

plt.tight_layout()
plt.show()

