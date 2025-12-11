#imports 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Time 
T = 5
dt = 0.0005
Nt = int(T / dt)

#init envi
L = np.array([4, 20]) # Lx = L [0] width, Ly = [1] height => Mesh size (µm)
Np = 1 #numbr of particules
radius_diameter = 10**(-2) #Radius of a particule (µm)

#init position and velocity
XY_start = np.array([0.2, L[1]/2])
Vp = np.array([20, 0])# in (µm/s)

Vp_dt = Vp * dt

Nt_index = np.arange(Nt)
XY_saved = np.array([XY_start[0] + Vp_dt[0]*Nt_index, XY_start [1] + Vp_dt[1]*Nt_index])
# print(XY_saved[0,:5])

# plot
# --- Figure and initial scatter ---
fig, ax = plt.subplots()

# Set domain to your physical box
ax.set_xlim(0, L[0])
ax.set_ylim(0, L[1])
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")

# One particle: start with empty data
scat = ax.scatter([], [], s=50)

# (Optional) nice aspect ratio
ax.set_aspect("equal", adjustable="box")

def update(frame):
    position_x = XY_saved [0, frame]
    position_y = XY_saved [1, frame]
    # set_offsets expects an array of shape (n_points, 2)
    scat.set_offsets([[position_x, position_y]])
    current_time = frame * dt
    ax.set_title(f"Particle animation (t = {current_time:.3f} s)")
    return scat,

# --- Create animation ---
ani = FuncAnimation(fig, update, frames=int(Nt/10),interval=10,blit=True)

plt.show()