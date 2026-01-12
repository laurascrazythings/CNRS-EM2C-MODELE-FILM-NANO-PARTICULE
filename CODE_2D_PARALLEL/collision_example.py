import numpy as np #np calculation
import matplotlib.pyplot as plt #plotting
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter #animation
from scipy.spatial import cKDTree


#functions to detect and solve collisions

def broad_detect(XY, d):

    """ Detect potential collision to prevent from testing every single pair using a KD-Tree for now

    Args:
        XY (array): Array of particle XY positions.
        d (float): superior to the Highest distance in one dt 

    Returns:
        array: Pairs of indices representing potential colliding particles. """

    tree = cKDTree(XY)  # Build a KD-Tree from particle positions
    pairs = tree.query_pairs(d)  # Find all pairs of particles within distance d
    
    return np.array(list(pairs),dtype = int)

def narrow_detect(Particle_test_pair, dt_left):
    """
    Test the collision of pairs of particles 
    
    Args:
        Particle_test_pair : array of the pair of particles collding 
    Returns:
        array: Pairs of colliding particle if they collide
        float: time of collision
    """
    v = Vp_local[Particle_test_pair[0], :] - Vp_local[Particle_test_pair[1], :] #velocity difference
    dif_pos = XY_local[Particle_test_pair[0], :] - XY_local[Particle_test_pair[1], :] #difference in position
    distance_collsion = 2 * Radius_particle #distance underwhich there is a collision, maybe to update later to enable different sizes of particles
    
    a = v @ v #the dot product will be positive if the direction is similar, if opposite it will be close to 0 or negative, meaning that whatever they do they wont cross
    if a < 1e-12:
        return None, None
    
    b = 2*(dif_pos @ v)
    c = (dif_pos @ dif_pos) - distance_collsion * distance_collsion
    
    if c < 0:
        return Particle_test_pair, -1 #return -1 to tell that the collision happened before
    
    if (dif_pos @ v) >= 0:
        return None, None
    
    disc = b * b -(4 * a * c)
    if disc <= 0.0:
        return None, None
    
    t_hit = (-b -np.sqrt(disc)) / (2 * a)
    if 0.0 <= t_hit <= dt_left:
        return Particle_test_pair, t_hit
    else:
        return None, None

def update_particles(XY_local, XY_local_update, Vp_local, Colliding_pair, t_collision):
    """
    update the particle velocity and its position at contact
    Args:
        Colliding_pair
        t_collision
        XY_local
        XY_local_update
        Vp_local
        
    Returns:
        XY_local
        Vp_local
    """
    #update the position of the colliding particles
    XY_local_update = XY_local + Vp_local * t_collision
    XY_local_update[Colliding_pair[0] , :] = XY_local[Colliding_pair[0] , :] + Vp_local[Colliding_pair[0], :] * t_collision
    XY_local_update[Colliding_pair[1] , :] = XY_local[Colliding_pair[1] , :] + Vp_local[Colliding_pair[1], :] * t_collision
    
    #Update the velocities of the particles
    distance_particles = XY_local_update[Colliding_pair[0] , :] - XY_local_update[Colliding_pair[1] , :] #distance vector of the particles
    dv = Vp_local[Colliding_pair[0], :] - Vp_local[Colliding_pair[1], :] #relative speed
    projected_distance = distance_particles @ distance_particles #||C1 - C2 ||^2
    if projected_distance > 1e-12:
        Vp_local[Colliding_pair[0], :] = Vp_local[Colliding_pair[0], :] - ((dv @ distance_particles) / projected_distance) * distance_particles
        Vp_local[Colliding_pair[1], :] = Vp_local[Colliding_pair[1], :] + ((dv @ distance_particles) / projected_distance) * distance_particles
    return XY_local_update, Vp_local
        
        
plt.clf()
T  = 20
dt = 0.05
L_total = np.array((50,50))
Nt = int(T/dt)
Num_particle = 9
XY_local = np.zeros((Num_particle,2))
XY_local[:, 0] = 1, 13, 49, 25.005, 6.0, 7.5, 9.0, 40, 40
XY_local[:, 1] = 1, 37, 1, 35, 45, 45, 45, 2, 10
Vp_local = np.zeros((Num_particle,2))
Vp_local[:, 0] = 2, 1, -2, 0.0, 1, 0, 0, 0, 0
Vp_local[:, 1] = 2, -1, 2, 0.0, 0, 0, 0, 0, -3
XY_local_update = XY_local.copy()
XY_local_saved = np.zeros((Nt, Num_particle, 2))
XY_local_saved[0,:,:] = XY_local.copy()
Radius_particle = 0.75

for t in range( 1, Nt):
    
    XY_local = XY_local_update.copy()
    dt_left = dt
    while dt_left > 0:
        d = np.max(abs(Vp_local))* dt_left * 2 + 2 * Radius_particle
        Particle_test = broad_detect(XY_local, d)
        if len(Particle_test) > 0:
            Colliding_pairs = []
            t_collisions = []
            for j in range(len(Particle_test)):
                Particle_colliding, t_hit = narrow_detect(Particle_test[j,:], dt_left)
                if t_hit is not None and t_hit >= 0.0 :
                    Colliding_pairs.append(Particle_colliding)
                    t_collisions.append(t_hit)
            if Colliding_pairs != []:
                idx = np.argmin(t_collisions)
                First_collision = Colliding_pairs[idx]#first colliding pair
                t_collision = t_collisions[idx] #time fo first collision
                XY_local_update, Vp_local = update_particles(XY_local, XY_local_update, Vp_local, First_collision, t_collision)
                XY_local = XY_local_update.copy()
                #Here I have the positions after the first hit
                dt_left = dt_left - t_collision
            else:
                XY_local_update = XY_local + Vp_local * dt_left
                dt_left = 0
        else:
            XY_local_update = XY_local + Vp_local * dt_left             
            dt_left = 0       
            
    XY_local_saved[t,:,:] = XY_local_update           
            



# plot
# --- Figure and initial scatter ---
fig, ax = plt.subplots()
# Set domain to your physical box
ax.set_xlim(0, L_total[0])
ax.set_ylim(0, L_total[1])
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")

N = XY_local_saved.shape[1]

# Example: fixed colors for each particle (2 particles here)
colors = np.array(["red", "blue", "green", "pink", "orange", "yellow", "brown", "black", "purple"])   # length N

scat = ax.scatter(
    XY_local_saved[0, :, 0], XY_local_saved[0, :, 1],
    s=50,
    c=colors
)
# #num particules - start with empty data
# scat = ax.scatter( np.zeros(1), np.zeros(1), s=50)
#define the title before
title = ax.set_title("Particle animation")

# --- choose video fps + how many sim steps per video frame ---
fps = 30
stride = max(1, int((1 / fps) / dt))   # sim steps per rendered frame
frames = range(0, Nt, stride)

#realistic aspect ratio
ax.set_aspect("equal", adjustable="box")

def update(frame):  #change to have n particules 
    positions = XY_local_saved[frame, :, :] 
    # set_offsets expects an array of shape (n_points, 2)
    scat.set_offsets(positions)
    current_time = frame * dt
    ax.set_title(f"Particle animation (t = {current_time:.3f} s)")

# --- Create animation ---  #change to have n particules 
ani = FuncAnimation( fig, update, frames = frames, interval=1000/fps, blit = False)

writer = FFMpegWriter(
fps=fps,
codec="libx264",
bitrate=1800)

ani.save("particle_animation_collision.mp4", writer=writer)

plt.close(fig)#not showing but saving

