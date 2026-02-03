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
        
def update_particles_aggregation(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par, Radius_particle, Num_Particules_end, Mass_particles, Cg_proc, Aggregate_set):
    """
    update the particle and its characteristics if the particles are agglomerating
    Args:
    
    Returns:

    """
    relative_index = np.zeros(2, dtype = int)
    print(Colliding_pair)
    relative_index[0] = Colliding_pair[0] % (Num_Particules_end)
    relative_index[1] = Colliding_pair[1] % (Num_Particules_end)
    i, j = relative_index
    Cgi = Cg_proc[i] #center of gravity of the first aggregate
    Cgj = Cg_proc[j] #center of gravity of the second aggregate
    numi = len(Aggregate_set[i]) #number of particles in the frist aggregate
    if numi == 0:
        numi = 1 #if the set is empty, there is the particle itself.
    numj = len(Aggregate_set[j]) #nmber of particles in the second aggregate
    if numj == 0:
        numj = 1 #if the set is empty, there is the particle itself.
    
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    cr = ((((Radius_particle[relative_index[0]] + Radius_particle[relative_index[1]]) *10**(-6))/0.15) * (-0.88)) + 0.78
    u_i = Vp_stack[Colliding_pair[0], :].copy()
    u_j = Vp_stack[Colliding_pair[1], :].copy()
    #update the position of the colliding particles
    XY_stack = XY_stack+ Vp_stack * t_collision * np.tile(Added_par[:,None], (9, 1)) #update all of the particles and then later change the one which collided
    XY_stack[Colliding_pair[0] , :] = XY_stack[Colliding_pair[0] , :].copy() + u_i * t_collision
    XY_stack[Colliding_pair[1] , :] = XY_stack[Colliding_pair[1] , :].copy() + u_j * t_collision
    #they should now be glued togther?
    merge = set(Aggregate_set[i]) #create a set that we populate and then use to replace the old set. easier
    merge.update(Aggregate_set[j])
    merge.update([i, j])
    cg_update = (Cgi * numi * Mass_particles[0]+ Cgj * numj * Mass_particles[0]) / (numi * Mass_particles[0] + numj * Mass_particles[0])
    V_new = (u_i * numi * Mass_particles[0] + u_j * numj * Mass_particles[0]) / (numi * Mass_particles[0] + numj * Mass_particles[0])
    
    for k in merge:
        Aggregate_set[k] = merge
        Cg_proc[k] = cg_update
        zone = zone_index(k)
        Vp_stack[zone * Num_Particules_end - 1, :] = V_new # average the velocity after impact? missing the e?
   
    return XY_stack, Vp_stack, Cg_proc, Aggregate_set
    
    
 

def interactions(Colliding_pair, A_h, Radius_particle, Mass_particle, Vp_stack, XY_stack, t_collision, Added_par, Num_particle_end):
    """
    find if the particles are aggregating or colliding
    
    """
    D0 = np.array(2)
    D0 = np.array([np.abs(XY_stack[Colliding_pair[0], 0] - XY_stack[Colliding_pair[1], 0]), np.abs(XY_stack[Colliding_pair[0], 1] - XY_stack[Colliding_pair[1], 1])])
    #take off the absolute, i JUST HAVE TO BE CONSISTENT WITHT EH WAU o AM DOIN GIT.
    if D0[0] == 0:
        D0[0] = 10**(-8)
    if D0[1] == 0:
        D0[1] = 10**(-8)
    #we need the distance in the form of an x y vecto r to get the proper forces, plug i in the right way?
    relative_index = np.zeros(2, dtype = int)
    relative_index[0] = Colliding_pair[0] % (Num_Particules_end)
    relative_index[1] = Colliding_pair[1] % (Num_Particules_end)
    #the following equation for adhering energy only works for particle of similar radiis but could be implemented otherwise using the following book:
    #https://www.eng.uc.edu/~beaucag/Classes/AdvancedMaterialsThermodynamics/Books/Jacob%20N.%20Israelachvili%20-%20Intermolecular%20and%20Surface%20Forces,%20Third%20Edition_%20Revised%20Third%20Edition-Academic%20Press%20(2011).pdf
    E_adh = np.zeros(2)
    E_adh = (A_h * Radius_particle[relative_index[0]]) / (12 * D0)
    F_adh = np.zeros(2)
    F_adh = E_adh / D0
    E_kin = np.zeros(2)
    E_kin = 0.5 * Mass_particle[relative_index[0]] * (Vp_stack[Colliding_pair[0],:]**(2)) + 0.5 *  Mass_particle[relative_index[1]] * (Vp_stack[Colliding_pair[1], :]**(2))
    F_kin = np.zeros(2)
    F_kin = E_kin / D0
    #try to implement a sort of flyby anomaly like in orbital mechanics?
    if F_adh < F_kin :
        update_particles_aggregation(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par, Radius_particle, Num_particle_end, Cg_proc)
    else:
        update_particles_collision(XY_stack,Vp_stack, Colliding_pair, t_collision, Added_par, Radius_particle, Mass_particle, Num_Particules_end)
    # if xxxx:
    #     Xy_stack, Vp_stack = update_particles_collision
    # else:
    #     update_particles_aggregation
            
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
Vp_local[:, 0] = 2, 1, -2, 0, 1, 0, 0, 0, 0
Vp_local[:, 1] = 2, -1, 2, 0, 0, 0, 0, 0, -3
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

