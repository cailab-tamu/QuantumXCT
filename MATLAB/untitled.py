import numpy as np
from qutip import Bloch, gates, basis
import matplotlib.pyplot as plt
# import imageio
import imageio.v2 as imageio

# Define initial states
states = {
    "|0>": basis(2, 0),
    "|1>": basis(2, 1),
    "|+>": (basis(2, 0) + basis(2, 1)).unit(),
    "|->": (basis(2, 0) - basis(2, 1)).unit()
}

# Colors for each state
colors = ['r', 'b', 'g', 'm']

# Rotation steps
angles = np.linspace(0, np.pi, 20)

# Store image frames for GIF
filenames = []

for i, angle in enumerate(angles):
    b = Bloch()
    b.point_color = colors
    b.vector_color = colors
    
    # Rotate each state about the X-axis
    for st in states.values():
        rotated = gates.ry(angle) * st
        b.add_states(rotated)
    
    # Save frame
    filename = f"bloch_frame_{i}.png"
    b.save(filename)
    plt.close('all')
    filenames.append(filename)

# Create animated GIF
with imageio.get_writer("y_gate_rotation.gif", mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("Animation saved as x_gate_rotation.gif")
