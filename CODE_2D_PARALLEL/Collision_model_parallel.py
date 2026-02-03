#By Laura Lambert Bocquet - lauralambertbocquet@gmail.com for CNRS, January 2026, Aymeric Vie
#The goal of this code is to have a cheap 3d numerical simulation of molecularar interactions at the micrometer level 
#For this it uses MPI or Message Per which allows the separation of the different cores of the cpu;
#It also uses non trivial coding sequences to allow for better O timings and tends to run as much as possible on O(log n ) or O(n log n)

#I am wanting to code a mpi code that can send one particule from one proc to another - 2D
from mpi4py import MPI #mpi
import numpy as np #np calculation
import matplotlib.pyplot as plt #plotting
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter #animation
from scipy.spatial import cKDTree #spatial tree to do a broad search


def zone_index(k):
    """
    from index of particle aggregating returns the zone the particle is in to allow for updating the velocity of the aggregate
    
    :param k: Description
    """
    if k in Index_par_local_set:
        zone = 8
    elif k in Index_par_ghost_right_set:
        zone = 0
    elif k in Index_par_ghost_left_set:
        zone = 1
    elif k in Index_par_ghost_up_set:
        zone = 2
    elif k in Index_par_ghost_down_set:
        zone = 3
    elif k in Index_par_ghost_up_right_set:
        zone = 4
    elif k in Index_par_ghost_down_right_set:
        zone = 5
    elif k in Index_par_ghost_down_left_set:
        zone = 6
    elif k in Index_par_ghost_up_right_set:
        zone = 7
    else :
        zone = None
    return zone

#functions to detect and solve collisions
#1 the first function is the broad phase
def broad_detect(XY, d):

    """ Detect potential collision to prevent from testing every single pair using a KD-Tree for now
    Args:
        XY (array): Array of particle XY positions.
        d (float): superior to the Highest distance in one dt 

    Returns:
        array: Pairs of indices representing potential colliding particles. """   
    mask = ~np.all(XY == 0.0, axis=1)   #mask the 0.0 positioned particles
    idx_valid = np.flatnonzero(mask) #keeps the indexes
    XY_non_zero = XY[mask] #position of the real particles
    tree = cKDTree(XY_non_zero)  # Build a KD-Tree from particle positions
    pairs = np.array(list(tree.query_pairs(d)),dtype = int) # Find all pairs of particles within distance d
    return idx_valid[pairs]

#2 the second function is the narrow phase
def narrow_detect(Particle_test_pair, dt_left, Vp_stack, XY_stack, Radius_molecule, Num_Particules_end):
    """
    Test the interaction of pairs of particles 
    
    Args:
        Particle_test_pair : array of the pair of particles collding 
        dt_left : float : dt left to the next dt
        Vp_stack : array of size 9* Number of particles and 2 dimensions: velocity of the particle
        XY_proc : array of size 9* Number of particles and 2 dimensions: position of the particle
    Returns:
        array: Pairs of colliding particle if they collide
        float: time of collision
    """
    relative_index = np.zeros(2, dtype = int)
    relative_index[0] = Particle_test_pair[0] % (Num_Particules_end)
    relative_index[1] = Particle_test_pair[1] % (Num_Particules_end)
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    v = Vp_stack[Particle_test_pair[0], :] - Vp_stack[Particle_test_pair[1], :] #velocity difference
    dif_pos = XY_stack[Particle_test_pair[0], :] - XY_stack[Particle_test_pair[1], :] #difference in position
    distance_collsion = Radius_molecule * 2 #distance underwhich there is a collision, maybe to update later to enable different sizes of particles
    a = v @ v #the dot product will be positive if the direction is similar, if opposite it will be close to 0 or negative, meaning that whatever they do they wont cross
    if a < 1e-12:
        return None, None
    
    b = 2*(dif_pos @ v)
    c = (dif_pos @ dif_pos) - distance_collsion * distance_collsion
    
    if c < 0:
        return Particle_test_pair, -1 #return -1 to tell that the collision happened before
    
    if (dif_pos @ v) >= 0:
        return None, None
    #solving for the time of impact can be transformed into a quadratic equation solving for time
    #discriminant
    disc = b * b -(4 * a * c)
    #if inferior to 0 then no physical solution
    if disc <= 0.0:
        return None, None
    #the first time of impact is 
    t_hit = (-b -np.sqrt(disc)) / (2 * a)
    if 0.0 <= t_hit <= dt_left:
        return Particle_test_pair, t_hit
    else:
        return None, None
    
#3 the third function is the update 
def update_particles_collision(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par, Radius_particle, Mass_one_particle, Num_Particules_end, Aggregate_set, Cg_stack):
    """
    update the particle velocity and its position at contact - collision case
    Args:
        Colliding_pair: Array of 2 particles, of 2 dimensions
        t_collision: float: time of collision
        XY_proc: array of size 9* Number of particles and 2 dimensions: position of the particle
        Vp_proc: array of size 9* Number of particles and 2 dimensions: velocity of the particle
        Added_par: array of size Number of particles and 1 dimensions: now if the particle is "sent" yet.
        Radius_molecule:
        Mass_molecule:
        
    Returns:
        XY_stack: array of size 9*Number of particles and 2 dimensions: position of the particle updated
        Vp_stack: array of size 9* Number of particles and 2 dimensions: velocity of the particle
        Aggregate_set
        Cg_proc
    """
    relative_index = np.zeros(2, dtype = int)
    relative_index[0] = Colliding_pair[0] % (Num_Particules_end)
    relative_index[1] = Colliding_pair[1] % (Num_Particules_end)
    i, j = relative_index
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    
    #calculate the coefficient of restitution - from the litterature I found that it was e = (d/0.15) * - 0,88 + 0.78. I am going to use this for now
    cr = ((((2 * Radius_molecule) *10**(-6))/0.15) * (-0.88)) + 0.78
    u_i = Vp_stack[Colliding_pair[0], :].copy()
    u_j = Vp_stack[Colliding_pair[1], :].copy()
    
    #get the mass of the 2 colliding sides
    numi = len(Aggregate_set[i]) #number of particles in the frist aggregate
    if numi == 0:
        numi = 1 #if the set is empty, there is the particle itself.
    numj = len(Aggregate_set[j]) #number of particles in the second aggregate
    if numj == 0:
        numj = 1 #if the set is empty, there is the particle itself.
        
    mass_i = Mass_one_particle * numi
    mass_j = Mass_one_particle * numj

    #update the position of the colliding particles
    XY_stack = XY_stack+ Vp_stack * t_collision * np.tile(Added_par[:,None], (9, 1)) #update all of the particles and then later change the one which collided
    XY_stack[Colliding_pair[0] , :] = XY_stack[Colliding_pair[0] , :].copy() + u_i * t_collision
    XY_stack[Colliding_pair[1] , :] = XY_stack[Colliding_pair[1] , :].copy() + u_j * t_collision
    #Update the velocities of the particles
    distance_particles = XY_stack[Colliding_pair[0] , :] - XY_stack[Colliding_pair[1] , :] #distance vector of the particles
    dv = u_i - u_j #relative speed
    projected_distance = distance_particles @ distance_particles #||C1 - C2 ||^2
    if projected_distance > 1e-12:
        Vp_stack[Colliding_pair[0], :] = (cr * mass_j * (u_j - u_i) + mass_i * u_i + mass_j * u_j)/ (mass_i + mass_j)
        Vp_stack[Colliding_pair[1], :] = (cr * mass_i * (u_i - u_j) + mass_i * u_i + mass_j * u_j)/ (mass_i + mass_j)
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set,

def update_particles_aggregation(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par, Radius_particle, Mass_one_particle, Num_Particules_end, Aggregate_set, Cg_stack):
    """
    update the particle and its characteristics if the particles are agglomerating
    Args:
    
    Returns:

    """
    relative_index = np.zeros(2, dtype = int)
    relative_index[0] = Colliding_pair[0] % (Num_Particules_end )
    relative_index[1] = Colliding_pair[1] % (Num_Particules_end )
    i, j = relative_index
    Cgi = Cg_stack[Colliding_pair[0], :] #center of gravity of the first aggregate
    Cgj = Cg_stack[Colliding_pair[1], :] #center of gravity of the second aggregate
    
    #get the mass of the 2 colliding sides
    numi = len(Aggregate_set[i]) #number of particles in the frist aggregate
    if numi == 0:
        numi = 1 #if the set is empty, there is the particle itself.
    numj = len(Aggregate_set[j]) #nmber of particles in the second aggregate
    if numj == 0:
        numj = 1 #if the set is empty, there is the particle itself.
        
    mass_i = Mass_one_particle * numi
    mass_j = Mass_one_particle * numj
    
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    cr = ((((2 * Radius_molecule) *10**(-6))/0.15) * (-0.88)) + 0.78
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
    cg_update = (Cgi * mass_i + Cgj * mass_j) / (mass_i + mass_j)
    V_new = (u_i * mass_i + u_j * mass_j) / (mass_i + mass_j)
    for k in merge:
        Aggregate_set[k] = merge.copy()
        zone = zone_index(k)
        if zone != None:
            Cg_stack[zone * Num_Particules_end + k, :] = cg_update.copy()
            Vp_stack[zone * Num_Particules_end + k, :] = V_new.copy() # average the velocity after impact? missing the e?
   
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set

#SCRIPT

#mpi init the domain of mpi - DO NOT TOUCH
comm = MPI.COMM_WORLD #INIT the mpi
size = comm.Get_size() #get the total num of processors
t0 = time.perf_counter() # register the time it was when it started

#2D cart : initialize the dim using cartesian mpi built in functions - DO NOT TOUCH
dimensions_proc = MPI.Compute_dims(size, [0, 0]) #computes the dimesions of each proc
cart = comm.Create_cart(dims = dimensions_proc, periods = [True, True] , reorder = True)
rank = cart.Get_rank() #get the processor rank
coordinates_x, coordinates_y = cart.Get_coords(rank) #coordinates in the respective proc
left, right = cart.Shift(direction = 0, disp = 1) #left right proc => direction 0
down, up = cart.Shift(direction = 1, disp = 1)# down up => direction 1
#the following code works because it is periodic for now. to change if non periodic
up_right = cart.Get_cart_rank([coordinates_x + 1, coordinates_y + 1])
down_right = cart.Get_cart_rank([coordinates_x + 1, coordinates_y - 1])
down_left = cart.Get_cart_rank([coordinates_x - 1 , coordinates_y - 1])
up_left = cart.Get_cart_rank([coordinates_x - 1, coordinates_y + 1])

#initialize the plot
plt.clf() 

#USER INIT
#time - TO SET

position = 0 #0 for auto and 1 for manual choice
T = 40# seconds to change - HERE
T_add_particles = 0.5 #time for which I add particles for - HERE
dt = 0.05 #delta t 
# save animation as a mp4? - To SET 
save_gif_animation = True 
#Mesh - TO SET
L_total = np.array([20, 20]) #Total Size in microm - HERE
#Particles - TO SET
Num_Particules = 400 #particles to start - HERE
Num_Particules_dt= 0 #particls added per second - HERE
#TiO2 properties - rutile for now
A_h = 6*10**(-20) #hamaker constant for rutile Tio2
Radius_molecule = 0.1 #radius of the particule in micrometer 
density = 4500000 #g/m3
Molar_mass = 79.9 # we can put the molecular mass in g/mol because we are only using this value for ratio calculation
Highest_velocity = 1 #velocity of the particle
Lowest_velocity = 0.01
Lowest_x_velocity = -1
Highest_x_velocity = 1
tau = 1.0 # relaxation time for Ornstein Uhlenbeck, chosen as 1 for now 
U_g = np.array([0, 1]) # mean gas velocity
B = 0.5 # noise intensity
wall = set(); #is there a wall, #0 for right, 1 for left, 2 for up, 3 for bottom, 4 for none
wall.add(2)
wall.add(3)
# wall.add(0)
# wall.add(1)
bounce = set() #is the wall bouncy?  #0 for right, 1 for left, 2 for up, 3 for bottom, 4 for none
# bounce.add(2)
# bounce.add(3)
# bounce.add(0)
# bounce.add(1)


#keep the periodicity but bounce if needed, the iff loops already exist to transfer the particle to the ghost; 

# init var - DO NOT TOUCH
Nt = int(T/ dt) #num of Iterations
Lx, Ly = L_total
Px, Py = dimensions_proc #neded to set the lines later 
#total number of particles by the end to create the right arrays and not resize(costly) - DO NOT TOUCH
Num_Particules_start = Num_Particules
Num_Particules_end = int(Num_Particules +  Num_Particules_dt * (T_add_particles /dt))

#define the charatcters of the particles
Volume = 4/3 * np.pi * (Radius_molecule*10**(-6))**3
Mass_one_particle = density * Volume * 10**(12) #picograms
Radius_one_particle = Radius_molecule

#particules init - DO NOT TOUCH
if rank == 1 : #do not overload proc 0, need the if so that the rand doesnt run for each proc 
    #velocity rand
    Vp = np.zeros((Num_Particules_end,2)) #init the velocity
    # if position == 0 random velocity, if position == 1 : choice of velocity
    if position == 0:
        Vp[:, 0] = np.random.uniform(low = Lowest_x_velocity, high = Highest_x_velocity, size = Num_Particules_end) # random x velocity, as the particles enter in the y direction , the whole set is random
        Vp[:Num_Particules, 1] = np.random.uniform(low = Lowest_velocity, high = Highest_velocity, size = Num_Particules)#the particle that start are random in y
        Vp[Num_Particules: Num_Particules_end, 1] = np.random.uniform(low = 0, high = Highest_velocity, size = Num_Particules_end - Num_Particules) #the particle that are added have a y direction that goes in the domain
    elif position == 1:
        Vp[:, 0] = 0, 0
        Vp[:, 1] = 1, 0.25
        
    #position rand - position after to ensure that the position the particules can be on arent in the 0 to buffer are where it would not be able to be sent periodically 
    XY_start = np.zeros((Num_Particules_end,2)) #particule start position 
    #the particle cannot touch at init, would break the code fro collision, they wouldnt collide basically
    maximum_velocity = np.sqrt(np.max(np.abs(Vp[:,0]))**(2) + np.max(np.abs(Vp[:,1]))**(2))
    #The particle can't start in the last bufferzone. it wont be able to be updated, there is only border control no spawn control taking into account the perodical condition
    Buffer_zone_width = np.array([maximum_velocity * dt* 2.01, (maximum_velocity + U_g[1]) * dt * 2.01]) #Buffer depends on the velocity of the particule, more than 2 to ensure that a particule pings twice 
    Starting_X = round((Buffer_zone_width[0] + Radius_molecule)/ (2 * Radius_molecule )) #starting possible position
    Starting_Y = round((Buffer_zone_width[1] + Radius_molecule)/ (2 * Radius_molecule))
    num_X = L_total[0] / (2 * Radius_molecule) #number of position possible
    left_over_x = num_X - int(num_X)
    num_Y = L_total[1] / (2 * Radius_molecule)
    left_over_y = num_Y - int(num_Y)
    
    add_num_Y = int(Buffer_zone_width[1] / (2 * Radius_molecule))#same but in the buffer or ghost zone really for the additional particles, they are generated at the start
    X_possible = np.arange(Starting_X, num_X + 1 - Starting_X) + 0.5 * left_over_x
    Y_possible = np.arange(Starting_Y, num_Y + 1 - Starting_Y) + 0.5 * left_over_y
    Add_Y_possible = np.arange(-add_num_Y, -1)
    X, Y = np.meshgrid(X_possible, Y_possible)
    X_add, Y_add = np.meshgrid(X_possible, Add_Y_possible)
    possible_position = (((np.stack((X, Y), axis = -1 ) * 2 * Radius_molecule) - Radius_molecule)  ).reshape(-1, 2)
    additional_position = (np.stack((X_add, Y_add), axis = -1 ) * 2 * Radius_molecule - Radius_molecule).reshape(-1, 2)

    #for the particles being added through the simulation time, we are going to initiate them at the same tie thant the others for better run time
    #they are going to be stocked where they spawn (at the bottom for now but can easily be changed)
    
    # #here the X and Y are reversed because when creating a mesh it is doing row column so YX so I am reshifting it
    # #Making pairs of position indexes to pull from
    # if position == 0 randomly positioned, if position == 1 : choice of position
    if position == 0:
        XY_start[:Num_Particules,:] = possible_position[np.random.choice(len(possible_position), size = Num_Particules, replace = False)]
        XY_start[Num_Particules:Num_Particules_end, :] = additional_position[np.random.choice(len(additional_position), size = Num_Particules_end - Num_Particules, replace = False)]
    elif position == 1:
        XY_start[:,0] = 10, 10
        XY_start[:,1] = 8, 9
        
    #register the transparent particles: the ones that are not displayed yet
    Added_par = np.zeros((Num_Particules_end), dtype = bool) 
    Added_par[:Num_Particules] = True   
else: 
    XY_start = None
    Vp = None
    Added_par = None
    Buffer_zone_width = None
    
#broadcast the values to the different procs
XY_start = cart.bcast(XY_start, root = 1)
Vp = cart.bcast(Vp, root = 1)
Added_par = cart.bcast(Added_par, root = 1)
Buffer_zone_width = cart.bcast(Buffer_zone_width, root = 1)
# #Buffer depends on the velocity of the particule, more than 2 to ensure that a particule pings twice 

#Each Proc's boundary - DO NOT TOUCH
Local_left = np.array([0]) # defining the left before being rewritten
Local_ghost_left = np.array([0])
Local_right = np.array([0]) # defining the right before being rewritten
Local_ghost_right = np.array([0])
Local_up = np.array([0])
Local_ghost_up = np.array([0])
Local_down = np.array([0])
local_ghost_down = np.array([0])

#initialize the start for each proc
Local_left = (coordinates_x/Px * Lx) # the principal zone starts
Local_ghost_left = Local_left - Buffer_zone_width[0] #the ghost zone starts before
Local_right = (coordinates_x + 1 )/Px * Lx #the prinicpal zone ends
Local_ghost_right = Local_right + Buffer_zone_width[0] #the ghost zone ends
Local_up= (coordinates_y + 1 )/Py * Ly #the prinicpal zone ends on the top
Local_ghost_up = Local_up + Buffer_zone_width[1] #the ghost zone ends
Local_down= (coordinates_y )/Py * Ly #the prinicpal zone ends on the bottom
Local_ghost_down = Local_down - Buffer_zone_width[1] #the ghost zone ends

# Particule position - to change - set to empty for now- pay attention in case some particules have a 00 position and 00 velocity might not work
XY_proc = np.zeros((9, Num_Particules_end, 2)) #creating a position array to replace the 9 different ones to deal with border collisions
Cg_proc = np.zeros((9, Num_Particules_end, 2)) #creating a center of gravity that depends on the position f the particles aggregated

##[:,0] : right; [:,1]: left; [:,2]: up; [:,3]: down [:,4] : up right; [:,5]: down right; [:,6]: down left; [:,7]: up left; [:. 8]: local
# Particule speeds- set to zeros for now
Vp_proc = np.zeros((9, Num_Particules_end, 2))
Aggregate_set = [set() for p in range(Num_Particules_end)]
#setting the particule index key
#num of particules per processor calculated thanks to len (Index)
#choice: there are one list and one set having the same value because it is more efficient to search in a set and reduces considerably the runtime 
Index_par_local = []
Index_par_local_set = set()
Index_par_ghost_left = []
Index_par_ghost_left_set = set()
Index_par_ghost_right = []
Index_par_ghost_right_set = set()
Index_par_ghost_up = []
Index_par_ghost_up_set = set ()
Index_par_ghost_down = []
Index_par_ghost_down_set = set ()
Index_par_ghost_up_right = []
Index_par_ghost_up_right_set = set ()
Index_par_ghost_down_right = []
Index_par_ghost_down_right_set = set ()
Index_par_ghost_down_left = []
Index_par_ghost_down_left_set = set ()
Index_par_ghost_up_left = []
Index_par_ghost_up_left_set = set ()
 
#Initialisation 
# of particules placement, keeping track of which particle is where (kind of like a master key)
for p in range(Num_Particules_end): 
    Position_x = XY_start[p, 0]
    Position_y = XY_start[p, 1]
    if Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up: #for now the processor has a particule if it is in
        Index_par_local.append(p) #know whch particle I have, helps not printing the same particle twice
        Index_par_local_set.add(p)
        XY_proc[8, p, :]= XY_start[p, :]
        Vp_proc[8, p, :]= Vp[p, :] #movment 
        Cg_proc[8, p, :] = XY_proc[8, p, :]
    elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up: #particule in the left ghost
        Index_par_ghost_left.append(p)
        Index_par_ghost_left_set.add(p)
        XY_proc[1, p, :] = XY_start[p, :]
        Vp_proc[1, p, :] = Vp[p, :] #movment 
        Cg_proc[1, p, :] = XY_proc[1, p, :]
    elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y < Local_up: #particule in the right ghost
        Index_par_ghost_right.append(p)
        Index_par_ghost_right_set.add(p)
        XY_proc[0, p, :] = XY_start[p, :]
        Vp_proc[0, p, :] = Vp[p, :] #movment 
        Cg_proc[0, p, :] = XY_proc[0, p, :]
    elif Local_left <= Position_x < Local_right and Local_up <= Position_y <= Local_ghost_up: #up ghost 
        Index_par_ghost_up.append(p)
        Index_par_ghost_up_set.add(p)
        XY_proc[2, p, :] = XY_start[p, :]
        Vp_proc[2, p, :] = Vp[p, :] #movment
        Cg_proc[2, p, :] = XY_proc[2, p, :] 
    elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down: #down ghost 
        Index_par_ghost_down.append(p)
        Index_par_ghost_down_set.add(p)
        XY_proc[3, p, :] = XY_start[p, :]
        Vp_proc[3, p, :] = Vp[p, :] #movment
        Cg_proc[3, p, :] = XY_proc[3, p, :]  
    elif Local_right <= Position_x <= Local_ghost_right and Local_up <= Position_y <= Local_ghost_up : #corners up right ghost
        Index_par_ghost_up_right.append(p)
        Index_par_ghost_up_right_set.add(p)
        XY_proc[4, p, :] = XY_start[p, :]
        Vp_proc[4, p, :] = Vp[p, :] #movment 
        Cg_proc[4, p, :] = XY_proc[4, p, :] 
    elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down : #corners down right ghost
        Index_par_ghost_down_right.append(p)
        Index_par_ghost_down_right_set.add(p)
        XY_proc[5, p, :] = XY_start[p, :]
        Vp_proc[5, p, :] = Vp[p, :] #movment 
        Cg_proc[5, p, :] = XY_proc[5, p, :] 
    elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down : #corners down left ghost
        Index_par_ghost_down_left.append(p)
        Index_par_ghost_down_left_set.add(p)
        XY_proc[6, p, :] = XY_start[p, :]
        Vp_proc[6, p, :] = Vp[p, :] #movment 
        Cg_proc[6, p, :] = XY_proc[6, p, :] 
    elif Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up : #corners up left ghost
        Index_par_ghost_up_left.append(p)
        Index_par_ghost_up_left_set.add(p)
        XY_proc[7, p, :] = XY_start[p, :]
        Vp_proc[7, p, :] = Vp[p, :]#movment
        Cg_proc[7, p, :] = XY_proc[7, p, :] 
    #because of the periodical condition, the ghosts of the borders is actually on the other side, it can cause some distubances in the t =0 because the particle dont fall into any categories    
         

#initiate the saving of information
#I wonder if I should print the particles in the ghost areas as well in different colors... For now, only the local xy will be printed
XY_master_saved = np.zeros((Nt, Num_Particules_end, 2)) #use at the end when merging the info
XY_local_saved = np.zeros((Nt, Num_Particules_end, 2)) #saving the positions of every particle of each processor in the time loop, will merge the info at the end
XY_local_saved[0,:,:] = XY_proc[8,:,:].copy()

# Boolean value that will keep track of if i have already sent a particule to another proc so 
# that it doesnt resend it, will be reset to false when the particule leaves the proc t enable rollback
Local_sent = np.zeros((Num_Particules_end, 8), dtype = bool) #[:,0] : right; [:,1]: left; [:,2]: up; [:,3]: down [:,4] : up right; [:,5]: down right; [:,6]: down left; [:,7]: up left


#time loop to update the map
for t in range(1, Nt+ 1):
    #update the velocities to match a brownian motion, we chose to go with the ornstein uhlenbeck implementation on speed
    xi = np.random.randn(2)  # Gaussian noise 
    #if Vp != 0.0 :  (doesnt work with aggregates bcause the velocities are dependant)
    #     Vp_proc[8,:,:] = Vp_proc[8,:,:] - ((Vp_proc[8,:,:] - U_g) * (dt/tau) + np.sqrt(B * dt) * xi)
    if rank == 0 and ((t * dt) % 2 ) == 0 :
        print(t*dt)
      #particles are getting added, add a certain number of particles, I have decided to do a boolean that changes value depending on the rate at which we add particle. The particle are "invisible" or at least wont move position until they are able to
    
    if t*dt <= T_add_particles:
        Added_par[Num_Particules : (Num_Particules + Num_Particules_dt )] = True
        Added_par_stack =  np.tile(Added_par[:,None], (9, 1))
        Num_Particules = Num_Particules + Num_Particules_dt       
        #update the transparent particules
    Particle_info_right = [] #list of particles to send to the right 
    Particle_info_left = [] #list of particles to send to the left
    Particle_info_up = [] # to up
    Particle_info_down = [] #to down
    Particle_info_up_right = [] #upright
    Particle_info_down_right = []#downright
    Particle_info_down_left = []#downleft
    Particle_info_up_left = []#upleft 
    
    if len(Index_par_local) > 0 or len(Index_par_ghost_left) > 0 or len(Index_par_ghost_right) > 0 or len(Index_par_ghost_down) > 0 or len(Index_par_ghost_up) > 0 or len(Index_par_ghost_up_right) > 0 or len(Index_par_ghost_down_right) > 0 or len(Index_par_ghost_down_left) > 0 or len(Index_par_ghost_up_left) :
        #there is one particule inside the whole processor realm so we update their position; if no particule in the area it should return 00
        dt_left = dt # set the dt left to the initial value before resolving collisions
        XY_stack = XY_proc.reshape(-1, 2)# stack the position values to allow for proimity checks
        Vp_stack = Vp_proc.reshape(-1, 2)
        Cg_stack = Cg_proc.reshape(-1, 2)
        
        
        while dt_left > 0 :
            d = np.sqrt((np.max(abs(Vp_stack[:,0])))**(2) + (np.max(abs(Vp_stack[:,1])))**(2)) * dt_left * 2 + 2 * Radius_molecule # 2* the distance the quickest particle can cover in 1dt
            Particle_test = broad_detect(XY_stack, d) #gets the short list of nearby particles in the whole proc thanks to the stack
            if len(Particle_test) > 0:
                Colliding_pairs = []
                t_collisions = []
                for j in range(len(Particle_test)):
                    Particle_colliding, t_hit = narrow_detect(Particle_test[j,:], dt_left, Vp_stack, XY_stack, Radius_molecule, Num_Particules_end)
                    if t_hit is not None and t_hit >= 0.0 :
                        Colliding_pairs.append(Particle_colliding)
                        t_collisions.append(t_hit)
                if Colliding_pairs != []:
                    idx = np.argmin(t_collisions)
                    First_collision = Colliding_pairs[idx]#first colliding pair
                    t_collision = t_collisions[idx] #time fo first collision
                    link = np.random.rand(1)
                    if link > 0.50:
                        XY_stack, Vp_stack, Cg_stack, Aggregate_set = update_particles_collision(XY_stack, Vp_stack, First_collision, t_collision, Added_par, Radius_molecule, Mass_one_particle, Num_Particules_end, Aggregate_set, Cg_stack)
                    else:
                        XY_stack, Vp_stack, Cg_stack, Aggregate_set = update_particles_aggregation(XY_stack, Vp_stack, First_collision, t_collision, Added_par, Radius_molecule, Mass_one_particle, Num_Particules_end, Aggregate_set, Cg_stack)
                    dt_left = dt_left - t_collision
                else:
                    XY_stack = XY_stack + Vp_stack * dt_left * Added_par_stack
                    dt_left = 0
            else:
                XY_stack = XY_stack + Vp_stack * dt_left * Added_par_stack             
                dt_left = 0 
              
        XY_proc = XY_stack.reshape(9, Num_Particules_end, 2)
        Vp_proc = Vp_stack.reshape(9, Num_Particules_end, 2)
        Cg_proc = Cg_stack.reshape(9, Num_Particules_end, 2)
        #now we have dealt with the transition from local to the ghosts, 
        # we have to deal with inter ghost and leaving interactions, for each ghost, there are 4 interactions possible.
        #right ghost interactions            
        for par_right in reversed(range(len(Index_par_ghost_right))): #in the ghost right area
            Index = Index_par_ghost_right[par_right] #index of the particule in the ghost right area to use for the xy vp which are indexed following the grand scheme to obliviate confusion
            Position_x = XY_proc[0,Index, 0]
            Position_y = XY_proc[0, Index, 1]
            #4 cases for now forward or backward or up or down
            #forward right to leave
            if Position_x > Local_ghost_right:
                #particule is leaving the ghost right area to nowhere as it is already sent to the next one
                XY_proc[0, Index, :] = [0, 0] #technically, I already sent it so no send
                Vp_proc[0, Index, :] = [0, 0]
                Cg_proc[0, Index, :] = [0, 0]
                Index_par_ghost_right.pop(par_right) #remove the particule from the count
                Index_par_ghost_right_set.discard(Index)
            #roll back right to local
            elif Position_x < Local_right and Local_down <= Position_y < Local_up:
                #the particule enters the local area and leaves the ghost right area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[0, Index, :].copy() #update the local position of the particule
                Vp_proc[8, Index, :] = Vp_proc[0, Index, :].copy() #Update the local speed of the particule
                Cg_proc[8, Index, :] = Cg_proc[0, Index, :].copy() #Update the local center of gravity
                XY_proc[0, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[0, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[0, Index, :] = [0, 0] #Remove the center of gravity
                Index_par_ghost_right.pop(par_right) #Index of the new particule added
                Index_par_ghost_right_set.discard(Index)
            #up to the ghost up-right 
            elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_up >= Position_y >= Local_up: 
                #the particle enters the up right ghost
                Index_par_ghost_up_right.append(Index) #add the particle to the list
                Index_par_ghost_up_right_set.add(Index)
                XY_proc[4, Index, :] = XY_proc[0, Index, :].copy() # update the local position
                Vp_proc[4, Index, :] = Vp_proc[0, Index, :].copy() #Update the local speed of the particule
                Cg_proc[4, Index, :] = Cg_proc[0, Index, :].copy() #Update the local center of gravity
                XY_proc[0, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[0, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[0, Index, :] = [0, 0] #Remove the center of gravity
                Index_par_ghost_right.pop(par_right) #Index of the new particule added
                Index_par_local_set.discard(Index)
            #down to the down ghost - down right
            elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down: 
                #the particle enters the down right ghost
                Index_par_ghost_down_right.append(Index) #add the particle to the list
                Index_par_ghost_down_right_set.add(Index)
                XY_proc[5, Index, :] = XY_proc[0, Index, :].copy() # update the local position
                Vp_proc[5, Index, :] = Vp_proc[0, Index, :].copy() #Update the local speed of the particule
                Cg_proc[5, Index, :] = Cg_proc[0, Index, :].copy() #Update the local center of gravity
                XY_proc[0, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[0, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[0, Index, :] = [0, 0] #Remove the center of gravity 
                Index_par_ghost_right.pop(par_right) #Index of the new particule added
                Index_par_local_set.discard(Index)
        
        #left ghost interactions    
        for par_left in reversed(range(len(Index_par_ghost_left))):
            Index = Index_par_ghost_left[par_left] #index of the particule in the ghost right area
            Position_x = XY_proc[1, Index, 0]
            Position_y = XY_proc[1, Index, 1]
            #4cases in 2d
            #forward left to local 
            if Position_x >= Local_left and Local_down <= Position_y < Local_up :
                #the particule enters the local area and leaves the ghost left area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[1, Index, :].copy() #update the local position of the particule
                Vp_proc[8, Index, :] = Vp_proc[1, Index, :].copy() #Update the local speed of the particule
                Cg_proc[8, Index, :] = Cg_proc[1, Index, :].copy() #Update the local center of gravity
                XY_proc[1, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[1, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[1, Index, :] = [0, 0] 
                Index_par_ghost_left.pop(par_left) #Index of the new particule added
                Index_par_ghost_left_set.discard(Index)
            #rollback left to leave
            elif Position_x < Local_ghost_left : 
                #particule is leaving the ghost right area 
                XY_proc[1, Index, :] = [0, 0] #technically, I already sent it so no send
                Vp_proc[1, Index, :] = [0, 0]
                Cg_proc[1, Index, :] = [0, 0]
                Index_par_ghost_left.pop(par_left) #remove the particule from the count
                Index_par_ghost_left_set.discard(Index)
            #down to down left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down:
                #particle enters the down left area
                Index_par_ghost_down_left.append(Index) #add the index to the list
                Index_par_ghost_down_left_set.add(Index)
                XY_proc[6, Index, :] = XY_proc[1, Index, :].copy() #update the local position of the particule
                Vp_proc[6, Index, :] = Vp_proc[1, Index, :].copy() #Update the local speed of the particule
                Cg_proc[6, Index, :] = Cg_proc[1, Index, :].copy() #Update the local center of gravity
                XY_proc[1, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[1, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[1, Index, :] = [0, 0]
                Index_par_ghost_left.pop(par_left) #Index of the new particule added
                Index_par_ghost_left_set.discard(Index)
            #up to up left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_up >= Position_y >= Local_up:
                #particle enters the up left area
                Index_par_ghost_up_left.append(Index) #add the index to the list
                Index_par_ghost_up_left_set.add(Index)
                XY_proc[7, Index, :] = XY_proc[1, Index, :].copy() #update the local position of the particule
                Vp_proc[7, Index, :] = Vp_proc[1, Index, :].copy() #Update the local speed of the particule
                Cg_proc[7, Index, :] = Cg_proc[1, Index, :].copy() #Update the local center of gravity
                XY_proc[1, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[1, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[1, Index, :] = [0, 0] 
                Index_par_ghost_left.pop(par_left) #Index of the new particule added
                Index_par_ghost_left_set.discard(Index)
        
        #up ghost interactions
        for par_up in reversed(range(len(Index_par_ghost_up))):
            Index = Index_par_ghost_up[par_up] #index of the particule in the ghost up area
            Position_x = XY_proc[2, Index, 0]
            Position_y = XY_proc[2, Index, 1]
            #4 cases in 2d
            #forward - up to up right
            if Local_right <= Position_x <= Local_ghost_right and Local_up <= Position_y <= Local_ghost_up:
                #enters the up right
                Index_par_ghost_up_right.append(Index) #add the index to the list
                Index_par_ghost_up_right_set.add(Index)
                XY_proc[4, Index, :] = XY_proc[2, Index, :].copy() #update the local position of the particule
                Vp_proc[4, Index, :] = Vp_proc[2, Index, :].copy() #Update the local speed of the particule
                Cg_proc[4, Index, :] = Cg_proc[2, Index, :].copy() #Update the local center of gravity
                XY_proc[2, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[2, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[2, Index, :] = [0, 0] 
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
            #backward - up to up left
            elif Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up:
                #enters the up left
                Index_par_ghost_up_left.append(Index) #add the index to the list
                Index_par_ghost_up_left_set.add(Index)
                XY_proc[7, Index, :] = XY_proc[2, Index, :].copy() #update the local position of the particule
                Vp_proc[7, Index, :] = Vp_proc[2, Index, :].copy() #Update the local speed of the particule
                Cg_proc[7, Index, :] = Cg_proc[2, Index, :].copy() #Update the local center of gravity
                XY_proc[2, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[2,  Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[2, Index, :] = [0, 0]
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
            #up - leaves
            elif Position_y > Local_ghost_up:
                #leaves the proc area
                XY_proc[2, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[2,  Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[2, Index, :] = [0, 0] 
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
            #down - local
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                #enters the local
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[2, Index, :].copy() #update the local position of the particule
                Vp_proc[8, Index, :] = Vp_proc[2,  Index, :].copy() #Update the local speed of the particule
                Cg_proc[8, Index, :] = Cg_proc[2, Index, :].copy() #Update the local center of gravity
                XY_proc[2, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[2, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[2, Index, :] = [0, 0]
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
        
        #down ghost interactions
        for par_down in reversed(range(len(Index_par_ghost_down))):
            Index = Index_par_ghost_down[par_down] #index of the particule in the ghost down area
            Position_x = XY_proc[3, Index, 0]
            Position_y = XY_proc[3, Index, 1]
            #4 cases in 2d
            #forward - down to down right
            if Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down:
                #enters the down right
                Index_par_ghost_down_right.append(Index) #add the index to the list
                Index_par_ghost_down_right_set.add(Index)
                XY_proc[5, Index, :] = XY_proc[3, Index, :].copy() #update the local position of the particule
                Vp_proc[5, Index, :] = Vp_proc[3, Index, :].copy() #Update the local speed of the particule
                Cg_proc[5, Index, :] = Cg_proc[3, Index, :].copy() #Update the local center of gravity
                XY_proc[3, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[3, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[3, Index, :] = [0, 0] 
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            #backward - down to down left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down:
                #enters the down left
                Index_par_ghost_down_left.append(Index) #add the index to the list
                Index_par_ghost_down_left_set.add(Index)
                XY_proc[6, Index, :] = XY_proc[3, Index, :].copy() #update the local position of the particule
                Vp_proc[6, Index, :] = Vp_proc[3, Index, :].copy() #Update the local speed of the particule
                Cg_proc[6, Index, :] = Cg_proc[3, Index, :].copy() #Update the local center of gravity
                XY_proc[3, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[3, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[3, Index, :] = [0, 0]
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            #down - leaves
            elif Position_y < Local_ghost_down:
                #leaves the proc area
                XY_proc[3, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[3, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[3, Index, :] = [0, 0] 
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            #up - local
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                #enters the local
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[3, Index, :].copy() #update the local position of the particule
                Vp_proc[8, Index, :] = Vp_proc[3, Index, :].copy() #Update the local speed of the particule
                Cg_proc[8, Index, :] = Cg_proc[3, Index, :].copy() #Update the local center of gravity
                XY_proc[3, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[3, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[3, Index, :] = [0, 0] 
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            
        #up right ghost interactions
        for par_up_right in reversed(range(len(Index_par_ghost_up_right))):
            Index = Index_par_ghost_up_right[par_up_right] #index of the particule in the ghost up right area
            Position_x = XY_proc[4, Index, 0]
            Position_y = XY_proc[4, Index, 1]
            #3 cases in 2d
            #forward or up - upright to leave
            # up or right - leaves
            if Position_y > Local_ghost_up or Position_x > Local_ghost_right:
                #leaves the proc area
                XY_proc[4, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[4, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[4, Index, :] = [0, 0]
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #down - ghost right
            elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y <= Local_up:
                #enters the ghost right
                Index_par_ghost_right.append(Index) #add the index to the list
                Index_par_ghost_right_set.add(Index)
                XY_proc[0, Index, :] = XY_proc[4, Index, :].copy() #update the local position of the particule
                Vp_proc[0, Index, :] = Vp_proc[4, Index, :].copy() #Update the local speed of the particule
                Cg_proc[0, Index, :] = Cg_proc[4, Index, :].copy()
                XY_proc[4, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[4, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[4, Index, :] = [0, 0] 
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #left - ghost up
            elif Local_left <= Position_x < Local_right and Local_up <= Position_y < Local_ghost_up:
                #enters the ghost up
                Index_par_ghost_up.append(Index) #add the index to the list
                Index_par_ghost_up_set.add(Index)
                XY_proc[2, Index, :] = XY_proc[4, Index, :].copy() #update the local position of the particule
                Vp_proc[2,  Index, :] = Vp_proc[4, Index, :].copy() #Update the local speed of the particule
                Cg_proc[2, Index, :] = Cg_proc[4, Index, :].copy()
                XY_proc[4, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[4, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[4, Index, :] = [0, 0] 
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[4, Index, :].copy()
                Vp_proc[8, Index, :] = Vp_proc[4, Index, :].copy()
                Cg_proc[8, Index, :] = Cg_proc[4, Index, :].copy()
                XY_proc[4, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[4, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[4, Index, :] = [0, 0] 
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
        
        #down right ghost interaction
        for par_down_right in reversed(range(len(Index_par_ghost_down_right))):
            Index = Index_par_ghost_down_right[par_down_right] #index of the particule in the ghost down right area
            Position_x = XY_proc[5, Index, 0]
            Position_y = XY_proc[5, Index, 1]
            #3 cases in 2d
            #forward or down - down right to leave
            # down or right - leaves
            if Position_y < Local_ghost_down or Position_x > Local_ghost_right:
                #leaves the proc area
                XY_proc[5, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[5, Index, :] = [0, 0] #Remove the speed of the particule 
                Cg_proc[5, Index, :] = [0, 0]
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #up - ghost right
            elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y < Local_up:
                #enters the ghost right
                Index_par_ghost_right.append(Index) #add the index to the list
                Index_par_ghost_right_set.add(Index)
                XY_proc[0, Index, :] = XY_proc[5, Index, :].copy() #update the local position of the particule
                Vp_proc[0, Index, :] = Vp_proc[5, Index, :].copy() #Update the local speed of the particule
                Cg_proc[0, Index, :] = Cg_proc[5, Index, :].copy()
                XY_proc[5, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[5, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[5, Index, :] = [0, 0] 
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #left - ghost down
            elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down:
                #enters the ghost down
                Index_par_ghost_down.append(Index) #add the index to the list
                Index_par_ghost_down_set.add(Index)
                XY_proc[3, Index, :] = XY_proc[5, Index, :].copy() #update the local position of the particule
                Vp_proc[3, Index, :] = Vp_proc[5, Index, :].copy() #Update the local speed of the particule
                Cg_proc[3, Index, :] = Cg_proc[5, Index, :].copy()
                XY_proc[5, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[5, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[5, Index, :] = [0, 0] 
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[5, Index, :].copy()
                Vp_proc[8, Index, :] = Vp_proc[5, Index, :].copy()
                Cg_proc[8, Index, :] = Cg_proc[5, Index, :].copy()
                XY_proc[5, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[5, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[5, Index, :] = [0, 0] 
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
                
        #down left ghost interactions
        for par_down_left in reversed(range(len(Index_par_ghost_down_left))):
            Index = Index_par_ghost_down_left[par_down_left] #index of the particule in the ghost down left area
            Position_x = XY_proc[6, Index, 0]
            Position_y = XY_proc[6, Index, 1]
            #3 cases in 2d
            #backward or down - down left to leave
            # down or left - leaves
            if Position_y < Local_ghost_down or Position_x < Local_ghost_left:
                #leaves the proc area
                XY_proc[6, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[6, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[6, Index, :] = [0, 0] 
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #up - ghost left
            elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up:
                #enters the ghost left
                Index_par_ghost_left.append(Index) #add the index to the list
                Index_par_ghost_left_set.add(Index)
                XY_proc[1, Index, :] = XY_proc[6, Index, :].copy() #update the local position of the particule
                Vp_proc[1, Index, :] = Vp_proc[6, Index, :].copy() #Update the local speed of the particule
                Cg_proc[1, Index, :] = Cg_proc[6, Index, :].copy()
                XY_proc[6, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[6, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[6, Index, :] = [0, 0] 
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #right - ghost down
            elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down:
                #enters the ghost down
                Index_par_ghost_down.append(Index) #add the index to the list
                Index_par_ghost_down_set.add(Index)
                XY_proc[3, Index, :] = XY_proc[6, Index, :].copy() #update the local position of the particule
                Vp_proc[3, Index, :] = Vp_proc[6, Index, :].copy() #Update the local speed of the particule
                Cg_proc[3, Index, :] = Cg_proc[6, Index, :].copy()
                XY_proc[6, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[6, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[6, Index, :] = [0, 0]  
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[6, Index, :].copy()
                Vp_proc[8, Index, :] = Vp_proc[6, Index, :].copy()
                Cg_proc[8, Index, :] = Cg_proc[6, Index, :].copy()
                XY_proc[6, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[6, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[6, Index, :] = [0, 0]  
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
        
        #up left ghost interactions
        for par_up_left in reversed(range(len(Index_par_ghost_up_left))):
            Index = Index_par_ghost_up_left[par_up_left] #index of the particule in the ghost up left area
            Position_x = XY_proc[7, Index, 0]
            Position_y = XY_proc[7, Index, 1]
            #3 cases in 2d
            #backward or down - down left to leave
            # up or left - leaves
            if Position_y > Local_ghost_up or Position_x < Local_ghost_left:
                #leaves the proc area
                XY_proc[7, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[7, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[7, Index, :] = [0, 0] 
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #down - ghost left
            elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up:
                #enters the ghost left
                Index_par_ghost_left.append(Index) #add the index to the list
                Index_par_ghost_left_set.add(Index)
                XY_proc[1, Index, :] = XY_proc[7, Index, :].copy() #update the local position of the particule
                Vp_proc[1, Index, :] = Vp_proc[7, Index, :].copy() #Update the local speed of the particule
                Cg_proc[1, Index, :] = Cg_proc[7, Index, :].copy()
                XY_proc[7, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[7, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[7, Index, :] = [0, 0] 
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #right - ghost up
            elif Local_left <= Position_x < Local_right and Local_up <= Position_y <= Local_ghost_up:
                #enters the ghost up
                Index_par_ghost_up.append(Index) #add the index to the list
                Index_par_ghost_up_set.add(Index)
                XY_proc[2, Index, :] = XY_proc[7, Index, :].copy() #update the local position of the particule
                Vp_proc[2,  Index, :] = Vp_proc[7, Index, :].copy() #Update the local speed of the particule
                Cg_proc[2, Index, :] = Cg_proc[7, Index, :].copy()
                XY_proc[7, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[7, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[7, Index, :] = [0, 0] 
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                XY_proc[8, Index, :] = XY_proc[7, Index, :].copy()
                Vp_proc[8, Index, :] = Vp_proc[7, Index, :].copy()
                Cg_proc[8, Index, :] = Cg_proc[7, Index, :].copy()
                XY_proc[7, Index, :] = [0, 0] #Remove the positon of the particule
                Vp_proc[7, Index, :] = [0, 0] #Remove the speed of the particule
                Cg_proc[7, Index, :] = [0, 0] 
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
        
        #update the local first if it is out of bounds 
        for par in reversed(range(len(Index_par_local))):
            Index = Index_par_local[par] #index in the xy, vp and other
            Position_x = XY_proc[8, Index, 0]
            Position_y = XY_proc[8, Index, 1]                
            #gathering the non covering cnditions to improve run time if and elifs that dont break the code basically
            #moving right -- technically but also called when moving backward
            if (Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and (Local_down + Buffer_zone_width[1]) <= Position_y < (Local_up - Buffer_zone_width[1]) and not(Local_right == L_total[0] and 0 in wall)): #xy >= ghost left du prochain #and not Local_sent[Index, 0]
                #The particule entered the left ghost zone of the right processor, 
                #we send it to the right proc if we havent yet
                Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                Local_sent[Index,0] = True
            #moving left
            elif (Local_left <= Position_x <= (Local_left + Buffer_zone_width[0]) and (Local_down + Buffer_zone_width[1]) <= Position_y < (Local_up - Buffer_zone_width[1]) and not(Local_left == 0 and 1 in wall)): 
                #we send it to the left proc if we havent yet
                Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, position, velocity
                Local_sent[Index, 1] = True
            #up
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and (Local_left + Buffer_zone_width[0]) <= Position_x < (Local_right - Buffer_zone_width[0]) and not(Local_up == L_total[1] and 2 in wall)): 
                #the particle has entered the down ghost zone of the up proc
                Particle_info_up.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
            #down
            elif (Local_down <= Position_y <= (Local_down + Buffer_zone_width[1]) and (Local_left + Buffer_zone_width[0]) <= Position_x < (Local_right - Buffer_zone_width[0]) and not(Local_down == 0 and 3 in wall)): 
                #the particle has entered the up ghost zone of the down particle
                Particle_info_down.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
            #up-right  
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and not(Local_right == L_total[0] and 0 in wall) and not(Local_up == L_total[1] and 2 in wall)):
                #enters the down left of the up right particle, but also the down of the up and the left of the right
                #down left of the up right
                Particle_info_up_right.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                Local_sent[Index, 4] = True
                #down of the up
                Particle_info_up.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
                #left of the right
                Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, position, velocity
                Local_sent[Index, 1] = True   
            #down-right
            elif (Local_down <= Position_y < (Local_down + Buffer_zone_width[1]) and Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and not(Local_right == L_total[0] and 0 in wall) and not(Local_down == 0 and 3 in wall)): 
                #enters the up left ghost of the down right proc but also the up of the down and the left of the right
                Particle_info_down_right.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                Local_sent[Index, 5] = True
                #up of the down
                Particle_info_down.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(),Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
                #left of the right
                Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, position, velocity
                Local_sent[Index, 1] = True
            #down-left
            elif (Local_down < Position_y < (Local_down + Buffer_zone_width[1]) and Local_left < Position_x <= (Local_left + Buffer_zone_width[0]) and not(Local_left == 0 and 1 in wall) and not(Local_down == 0 and 3 in wall)):
                #enters the up right ghost of the down left particle but also in the up of the down and the right of the left
                Particle_info_down_left.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                Local_sent[Index, 6] = True
                #up of the down
                Particle_info_down.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
                #right of the left
                Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                Local_sent[Index,0] = True   
            #up left
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and Local_left < Position_x <= (Local_left + Buffer_zone_width[0]) and not(Local_left == 0 and 1 in wall) and not(Local_up == L_total[1] and 2 in wall)):
                #enters the down right ghost of the up left particle but also down of ht ep and right of the left
                Particle_info_up_left.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                Local_sent[Index, 7] = True
                #down of the up 
                Particle_info_up.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
                #right of the left
                Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                Local_sent[Index,0] = True 
                
            #particule is leaving the local area for a ghost position    
            #nominally in the same proc it should only be in one space so elifs work because only one condition may be true.
            #local to right ghost
            if Position_x >= Local_right and Local_up > Position_y >= Local_down:
                #if right is a wall then change the position and velocity if not then pass it on to the right ghost 
                if Local_right != L_total[0] or 0 not in wall:  
                    #the particule left the main local proc area, enters the right ghost area 
                    Index_par_ghost_right.append(Index)#add the index to the end of the list  
                    Index_par_ghost_right_set.add(Index)        
                    XY_proc[0, Index,:] = XY_proc[8, Index, :].copy() #associate the local Position with the ghost         
                    Vp_proc[0, Index, :] = Vp_proc[8, Index, :].copy() #associate the local speed with the ghost
                    Cg_proc[0, Index, :] = Cg_proc[8, Index, :].copy() 
                    if Local_sent[Index, 0] == False: #saved a particle that changed direction while being in the boundary area
                        Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,0] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_right)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, :] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_right)
            #local to left ghost 
            elif Position_x < Local_left and Local_down <= Position_y < Local_up :
                #if left is a wall then change the position and velocity if not then pass it on to the left ghost 
                if Local_left != 0 or 1 not in wall:
                    #the particule left the main local proc area, enters the left ghost area    
                    Index_par_ghost_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_left_set.add(Index)         
                    XY_proc[1, Index,:] = XY_proc[8, Index, :].copy() #associate the local position with the ghost         
                    Vp_proc[1, Index, :] = Vp_proc[8, Index, :].copy() #associate the local speed with the ghost
                    Cg_proc[1, Index, :] = Cg_proc[8, Index, :].copy() 
                    if Local_sent[Index, 1] == False:
                        Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,1] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_left)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, 0] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_left)
            #local to up 
            elif(Position_y >= Local_up and Local_left <= Position_x < Local_right):
                #if up is a wall then change the position and velocity if not then pass it on to the up ghost 
                if Local_up != L_total[1] or 2 not in wall:
                    #the particle enters the ghost up area and leaves the local
                    Index_par_ghost_up.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_set.add(Index)
                    XY_proc[2, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the up ghost
                    Vp_proc[2,  Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost up speed for now( no collisions)
                    Cg_proc[2, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 2] == False:
                        Particle_info_up.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,2] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else :
                    if 2 in bounce:
                        XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_up)  #set the position to the changed position after impact
                        Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                        Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_up)
                    else: 
                        XY_proc[8, Index, 1] = Local_up
                        Vp_proc[8, Index, :] = [0, 0]
                        Cg_proc[8, Index, :] = XY_proc[8, Index, :]
            #local to down
            elif(Position_y < Local_down and Local_left <= Position_x < Local_right):
                #if down is a wall then change the position and velocity if not then pass it on to the down ghost 
                if Local_down != 0 or 3 not in wall:
                    #the particle enters the ghost down area and leaves the local
                    Index_par_ghost_down.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_set.add(Index)
                    XY_proc[3, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the down ghost
                    Vp_proc[3, Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost down speed for now( no collisions)
                    Cg_proc[3, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 3] == False:
                        Particle_info_down.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,3] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_down)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                    Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_down)
            #local to up right    
            elif Position_x >= Local_right and Local_up <= Position_y <= Local_ghost_up:
                #if up and right are a wall then change the position and velocity if not then pass it on to the up right ghost 
                if (Local_up != L_total[1] or 2 not in wall) and (Local_right != L_total[0] or 0 not in wall):
                    #particle enters the ghost up right area
                    Index_par_ghost_up_right.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_right_set.add(Index)
                    XY_proc[4, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the ghost
                    Vp_proc[4, Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost speed for now( no collisions)
                    Cg_proc[4, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 4] == False:
                        Particle_info_up_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index, 4] = True
                        #down of the up
                        Particle_info_up.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                        Local_sent [Index, 2] = True
                        #left of the right
                        Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, position, velocity
                        Local_sent[Index, 1] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_up == L_total[1] and 2 in wall): 
                    if 2 in bounce:
                        XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_up)  #set the position to the changed position after impact
                        Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                        Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_up)
                    else:
                        XY_proc[8, Index, 1] = Local_up
                        Vp_proc[8, Index, :] = [0, 0]
                        Cg_proc[8, Index, :] = XY_proc[8, Index, :]
                        
                if (Local_right == L_total[0] and 0 in wall):
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_right)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, 0] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_right)
                    
            #local to down right
            elif Position_x >= Local_right and Local_ghost_down <= Position_y < Local_down:
                #if down and right are a wall then change the position and velocity if not then pass it on to the down right ghost 
                if (Local_down != 0 or 3 not in wall) and (Local_right != L_total[0] or 0 not in wall):
                    #particle enters the ghost down right area
                    Index_par_ghost_down_right.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_right_set.add(Index)
                    XY_proc[5, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the ghost
                    Vp_proc[5, Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost speed for now( no collisions)
                    Cg_proc[5, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 5] == False:
                        #enters the up left ghost of the down right proc but also the up of the down and the left of the right
                        Particle_info_down_right.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                        Local_sent[Index, 5] = True
                        #up of the down
                        Particle_info_down.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                        Local_sent [Index, 3] = True
                        #left of the right
                        Particle_info_right.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, position, velocity
                        Local_sent[Index, 1] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_down == 0 and 3 in wall): #only the right bounces
                    XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_down)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                    Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_down)
                if (Local_right == L_total[0] and 0 in wall):
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_right)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, 0] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_right)
            #local to down left
            elif (Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down):
                #if down and left are a wall then change the position and velocity if not then pass it on to the down left ghost 
                if (Local_down != 0 or 3 not in wall) and (Local_left != 0 or 1 not in wall):
                    #particle enters the down left area
                    Index_par_ghost_down_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_left_set.add(Index)
                    XY_proc[6, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the ghost
                    Vp_proc[6, Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost speed for now( no collisions)
                    Cg_proc[6, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 6] == False:
                        #enters the up right ghost of the down left particle but also in the up of the down and the right of the left
                        Particle_info_down_left.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                        Local_sent[Index, 6] = True
                        #up of the down
                        Particle_info_down.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                        Local_sent [Index, 3] = True
                        #right of the left
                        Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,0] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_down == 0 and 3 in wall):
                    XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_down)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                    Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_down)
                if (Local_left == 0 and 1 in wall) :
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_left)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, 0] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_left)
            #local to up left
            elif (Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up):
                #if up and left are a wall then change the position and velocity if not then pass it on to the up left ghost 
                if (Local_up != L_total[1] or 2 not in wall) and (Local_left != 0 or 1 not in wall):
                    #particle enters the up left area
                    Index_par_ghost_up_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_left_set.add(Index)
                    XY_proc[7, Index, :] = XY_proc[8, Index, :].copy()#associate the updated local position with the ghost
                    Vp_proc[7, Index, :] = Vp_proc[8, Index, :].copy() #local speed => ghost speed for now( no collisions)
                    Cg_proc[7, Index, :] = Cg_proc[8, Index, :].copy()
                    if Local_sent[Index, 7] == False:
                        #enters the down right ghost of the up left particle but also down of ht ep and right of the left
                        Particle_info_up_left.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy()))
                        Local_sent[Index, 7] = True
                        #down of the up 
                        Particle_info_up.append((Index, XY_proc[8, Index, :].copy(), Vp_proc[8, Index, :].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) #index, position, velocity
                        Local_sent [Index, 2] = True
                        #right of the left
                        Particle_info_left.append((Index, XY_proc[8, Index,:].copy(), Vp_proc[8, Index,:].copy(), Aggregate_set[Index].copy(), Cg_proc[8, Index, :].copy())) # index, Position, velocity
                        Local_sent[Index,0] = True
                    XY_proc[8, Index, :] = [0, 0] #set the local to 00 
                    Vp_proc[8, Index, :] = [0, 0] # set the local back to 00
                    Cg_proc[8, Index, :] = [0, 0]
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if Local_up == L_total[1] and 2 in wall:
                    if 2 in bounce:
                        XY_proc[8, Index, 1] = XY_proc[8, Index, 1] - (XY_proc[8, Index, 1] - Local_up)  #set the position to the changed position after impact
                        Vp_proc[8, Index, 1] = - Vp_proc[8, Index, 1] #reverse the speed in the y direction
                        Cg_proc[8, Index, 1] = Cg_proc[8, Index, 1] - (Cg_proc[8, Index, 1] - Local_up)
                    else:
                        XY_proc[8, Index, 1] = Local_up
                        Vp_proc[8, Index, :] = [0, 0]
                        Cg_proc[8, Index, :] = XY_proc[8, :, :]
                    
                if Local_left == 0 or 1 in wall:
                    XY_proc[8, Index, 0] = XY_proc[8, Index, 0] - (XY_proc[8, Index, 0] - Local_left)  #set the position to the changed position after impact
                    Vp_proc[8, Index, 0] = - Vp_proc[8, Index, 0] #reverse the speed in the x direction
                    Cg_proc[8, Index, 0] = Cg_proc[8, Index, 0] - (Cg_proc[8, Index, 0] - Local_left)
                    
            #we are reseting the booleans at the end to ensure that the other conditions are viewed    
            if ((Position_x < (Local_right - Buffer_zone_width[0]) or Position_x >= Local_right or Position_y >= Local_up or Position_y < Local_down) and Local_sent[Index, 0]): #if it leaves on the other side back to false
                #print("The particule has left the proc")
                Local_sent[Index, 0] = False
            if ((Position_x > (Local_left + Buffer_zone_width[0]) or Position_x < Local_left or Position_y >= Local_up or Position_y < Local_down )and Local_sent[Index, 1]): #reset the boolean to ensure the particule can be sent again
                #print("The particule has left the proc") 
                Local_sent[Index, 1] = False
            if ((Position_y < (Local_up - Buffer_zone_width[1]) or Position_y >= Local_up or Position_x >= Local_right or Position_x < Local_left) and Local_sent[Index, 2]): 
                #reset the boolean
                Local_sent [Index, 2] = False
            if ((Position_y > (Local_down + Buffer_zone_width[1]) or Position_y < Local_down or Position_x >= Local_right or Position_x < Local_left ) and Local_sent[Index, 3]):
                #reset the boolean
                Local_sent [Index, 3] = False
            if (Position_y < (Local_up - Buffer_zone_width[1]) or Position_y >= Local_up or Position_x >= Local_right or Position_x < (Local_right - Buffer_zone_width[0])) and Local_sent[Index, 4]:
                #reset the boolean
                Local_sent [Index, 4] = False 
            if (Position_y > (Local_down + Buffer_zone_width[1]) or Position_y < Local_down or Position_x < (Local_right - Buffer_zone_width[0]) or Position_x >= Local_right) and Local_sent[Index, 5]:
                #resets the boolean
                Local_sent[Index, 5] = False
            if (Position_y > (Local_down + Buffer_zone_width[1]) or Position_y < Local_down or Position_x > (Local_left + Buffer_zone_width[0]) or Position_x < Local_left) and Local_sent[Index,6]:
                #resets the boolean
                Local_sent[Index, 6] = False
            if (Position_y < (Local_up - Buffer_zone_width[1]) or Position_y >= Local_up or Position_x > (Local_left + Buffer_zone_width[0]) or Position_x < Local_left) and Local_sent[Index,7]:
                #resets the boolean
                Local_sent[Index, 7] = False
 
    #do the comms now, so that it send the whole list
    incoming_from_left = cart.sendrecv( sendobj = Particle_info_right, dest = right, sendtag = 0, source = left, recvtag = 0)
    #sending a particule to the right, the tag helps identifying hte different sends (if it is going left or right)
    incoming_from_right = cart.sendrecv( sendobj = Particle_info_left, dest = left, sendtag = 1, source = right, recvtag = 1)
    #sending a particule to the left, tag = 1 for right to left
    incoming_from_up = cart.sendrecv( sendobj = Particle_info_down, dest = down, sendtag = 2, source = up, recvtag = 2)
    #send a particle to down, 
    incoming_from_down = cart.sendrecv( sendobj = Particle_info_up, dest = up, sendtag = 3, source = down, recvtag = 3)
    #send a particle to up
    incoming_from_up_right = cart.sendrecv( sendobj = Particle_info_down_left, dest = down_left, sendtag = 4, source = up_right, recvtag = 4)
    #send a particle to down left
    incoming_from_down_right = cart.sendrecv( sendobj = Particle_info_up_left, dest = up_left, sendtag = 5, source = down_right, recvtag = 5)
    #send a particle from up left
    incoming_from_down_left = cart.sendrecv(sendobj = Particle_info_up_right, dest = up_right, sendtag = 6, source = down_left, recvtag = 6)
    #send a particle from up right
    incoming_from_up_left = cart.sendrecv(sendobj = Particle_info_down_right, dest = down_right, sendtag = 7, source = up_left, recvtag = 7)
    #SEND a particle from down right
    
    
    #how to deal with the new data
    #incoming from left = ghost left
    if incoming_from_left is None: #if null
        incoming_from_left = [] #set to empty to be able to loop on it without errors
    # elif incoming_from_left != []:
    #     print("incoming from left: ", incoming_from_left)
    for Index, pos, vel, aggregate, c_grav in incoming_from_left: #for all the different particles
        #add to ghost left
        if coordinates_x == 0:
            pos[0] = pos[0] - L_total[0] #if it comes from the last proc, the position is equal to 
        if Index not in Index_par_ghost_left_set:
            Index_par_ghost_left.append(Index) #add the particle index
            Index_par_ghost_left_set.add(Index)
            XY_proc[1, Index] = pos #associate the postion
            Vp_proc[1,  Index] = vel #and velocity
            Aggregate_set[Index] = aggregate
            Cg_proc[1, Index] = c_grav
        elif not(np.allclose(Vp_proc[1, Index], vel)):
            XY_proc[1, Index] = pos
            Vp_proc[1,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[1, Index] = c_grav

    #incoming from right = ghost right
    if incoming_from_right is None: #if null
        incoming_from_right = [] #set to empty to be able to loop on it without errors
    # elif incoming_from_right != [] :
    #      print("for rank : ", rank,"incoming from right : ", incoming_from_right)           
    for Index, pos, vel, aggregate, c_grav in incoming_from_right: #for all the different particles
        #add to ghost b
        if coordinates_x == (dimensions_proc[0] - 1) :
            pos[0] = pos[0] + L_total[0] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_right_set:
            Index_par_ghost_right.append(Index)
            Index_par_ghost_right_set.add(Index)
            XY_proc[0, Index] = pos
            Vp_proc[0, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[0, Index] = c_grav
        elif not(np.allclose(Vp_proc[0, Index], vel)):
            XY_proc[0, Index] = pos
            Vp_proc[0,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[0, Index] = c_grav
    
    #incoming from up = ghost up
    if incoming_from_up is None:
        incoming_from_up = []
    # elif incoming_from_up != []:
    #     print("incoming from up: ", incoming_from_up)
    for Index, pos, vel, aggregate, c_grav in incoming_from_up:
        if coordinates_y == (dimensions_proc[1] - 1):
            pos[1] = pos[1] + L_total[1]
        if Index not in Index_par_ghost_up_set:
            Index_par_ghost_up.append(Index)
            Index_par_ghost_up_set.add(Index)
            XY_proc[2, Index] = pos
            Vp_proc[2,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[2, Index] = c_grav
        elif not(np.allclose(Vp_proc[2, Index], vel)):
            XY_proc[2, Index] = pos
            Vp_proc[2,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[2, Index] = c_grav
    
    #incoming from down = ghost down
    if incoming_from_down is None:
        incoming_from_down = []
    # elif incoming_from_down != []:
    #      print("for rank : ", rank,"incoming from down : ", incoming_from_down)
    for Index, pos, vel, aggregate, c_grav in incoming_from_down:
        if coordinates_y == 0:
            pos[1] = pos[1] - L_total[1]
        if Index not in Index_par_ghost_down_set:
            Index_par_ghost_down.append(Index)
            Index_par_ghost_down_set.add(Index)
            XY_proc[3, Index] = pos
            Vp_proc[3, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[3, Index] = c_grav
        elif not(np.allclose(Vp_proc[3, Index], vel)):
            XY_proc[3, Index] = pos
            Vp_proc[3,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[3, Index] = c_grav
    
    #incoming from up right = ghost up right
    if incoming_from_up_right is None:
        incoming_from_up_right = []
    # elif incoming_from_up_right != []:
    #     print("incoming from up right: ", incoming_from_up_right)
    for Index, pos, vel, aggregate, c_grav in incoming_from_up_right:
        if coordinates_x == (dimensions_proc[0] - 1):
            pos[0] = pos[0] + L_total[0]
        if coordinates_y == (dimensions_proc[1] - 1):
            pos[1] = pos[1] + L_total[1]
        if Index not in Index_par_ghost_up_right_set:
            Index_par_ghost_up_right.append(Index)
            Index_par_ghost_up_right_set.add(Index)
            XY_proc[4, Index] = pos
            Vp_proc[4, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[4, Index] = c_grav
        elif not(np.allclose(Vp_proc[4, Index], vel)):
            XY_proc[4, Index] = pos
            Vp_proc[4,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[4, Index] = c_grav
    
    #incoming from down right = ghost down right
    if incoming_from_down_right is None:
        incoming_from_down_right = []
    # elif incoming_from_down_right != []:
    #      print("incoming from down right: ", incoming_from_down_right)
    for Index, pos, vel, aggregate, c_grav in incoming_from_down_right:
        if coordinates_x == (dimensions_proc[0] - 1) :
            pos[0] = pos[0] + L_total[0]
        if coordinates_y == 0:
            pos[1] = pos[1] - L_total[1]
        if Index not in Index_par_ghost_down_right_set:
            Index_par_ghost_down_right.append(Index)
            Index_par_ghost_down_right_set.add(Index)
            XY_proc[5, Index] = pos
            Vp_proc[5, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[5, Index] = c_grav
        elif not(np.allclose(Vp_proc[5, Index], vel)):
            XY_proc[5, Index] = pos
            Vp_proc[5,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[5, Index] = c_grav
    
    #incoming from down left = ghost down left
    if incoming_from_down_left is None:
        incoming_from_down_left = []
    # elif incoming_from_down_left != []:
    #     print("incoming from down left: ", incoming_from_down_left)
    for Index, pos, vel, aggregate, c_grav in incoming_from_down_left:
        if coordinates_x == 0 :
            pos[0] = pos[0] - L_total[0]
        if coordinates_y == 0:
            pos[1] = pos[1] - L_total[1]
        if Index not in Index_par_ghost_down_left_set:
            Index_par_ghost_down_left.append(Index)
            Index_par_ghost_down_left_set.add(Index)
            XY_proc[6, Index] = pos
            Vp_proc[6, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[6, Index] = c_grav
        elif not(np.allclose(Vp_proc[6, Index], vel)):
            XY_proc[6, Index] = pos
            Vp_proc[6,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[6, Index] = c_grav
            
    #incoming from up left = ghost up left
    if incoming_from_up_left is None:
        incoming_from_up_left = []
    # elif incoming_from_up_left != []:
    #      print("incoming from up left: ", incoming_from_up_left)
    for Index, pos, vel, aggregate, c_grav in incoming_from_up_left:
        if coordinates_x == 0:
            pos[0] = pos[0] - L_total[0]
        if coordinates_y == (dimensions_proc[1] - 1):
            pos[1] = pos[1] + L_total[1]
        if Index not in Index_par_ghost_up_left_set:
            Index_par_ghost_up_left.append(Index)
            Index_par_ghost_up_left_set.add(Index)
            XY_proc[7, Index] = pos
            Vp_proc[7, Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[7, Index] = c_grav
        elif not(np.allclose(Vp_proc[7, Index], vel)):
            XY_proc[7, Index] = pos
            Vp_proc[7,  Index] = vel
            Aggregate_set[Index] = aggregate
            Cg_proc[7, Index] = c_grav
            
    XY_local_saved[t - 1, :, :] = XY_proc[8,:,:].copy() #save the updated values 
    
#saving the values of the local particles 
if rank == 0: #the proc 0 will receive all of the other locally saved positoins of particles
    XY_master_saved = XY_local_saved.copy() #first let's say that the master positions is proc 0s and add the different position if dif than 00
    #I want to add the values dif than 0 from XY_source saved to the XY_master_saved to save the particle movment 
    for source in range(1,size): #go thru all of the proc
        XY_source_saved = cart.recv(source = source) #receive the send from the other proc
        for t in range (Nt): #going through all of the time dimension
            for i in range(Num_Particules_end): #going through all of the particules
                if not np.allclose(XY_source_saved[t, i, :], 0.0):#if dif than null which is 00 for now, using this instead of direct comparaison beause there might be floats
                    if np.allclose(XY_master_saved[t, i, :], 0.0) : #if the master value is set to 00 for now 
                        XY_master_saved[t, i, :] = XY_source_saved[t, i, :].copy() #replace the value in the master for the source value,
                    elif not np.allclose(XY_master_saved[t, i, :], XY_source_saved[t, i, :]): # check if 2 dif value dif than 00 exist for 2 dif proc
                        dx = XY_master_saved[t, i, 0] - XY_source_saved[t, i, 0]
                        dy = XY_master_saved[t, i, 1] - XY_source_saved[t, i, 1]
                        remainer_x = dx % L_total[0] 
                        remainer_y = dy % L_total[1]
                        if not (np.allclose(remainer_x, 0.0) or np.allclose(remainer_x, L_total[0])) and (np.allclose(remainer_y,0.0) or np.allclose(remainer_y,L_total[1])) :
                            print("The position ",XY_source_saved[t, i, :] , " for particule ", i, " in proc: ", source, " is different than in the master at t = ", t, "\n", " which is : ", XY_master_saved[t, i, :])
    zero = ~np.all(XY_master_saved[Nt-1,:,:] == 0., axis=1)
    XY_count = XY_master_saved[Nt-1, zero]
    Num_Particules_end_count = len(XY_count)
    dif_num = abs(Num_Particules_end_count - Num_Particules_end)                               
    if  dif_num != 0:
        print("OH NO, there are ", dif_num," missing particles ")       
else :
    cart.send(XY_local_saved, dest = 0)

#WAIT ON THE OTHER PROC TO FINISH
comm.Barrier()
t1 = time.perf_counter()

def radius_to_s(ax, Radius):
    # two points separated by R in x-direction
    p1 = ax.transData.transform((0, 0))
    p2 = ax.transData.transform((Radius, 0))

    # pixel distance
    pixel_radius = abs(p2[0] - p1[0])

    # pixels → points (1 pt = 1/72 inch)
    fig = ax.figure
    dpi = fig.dpi
    point_radius = pixel_radius * 72.0 / dpi

    # scatter size is area in points^2
    return np.pi * point_radius**2

#printing and ploting
if rank == 0:
    
    # plot
    # --- Figure and initial scatter ---
    fig, ax = plt.subplots()
    # Set domain to your physical box
    ax.set_xlim(0, L_total[0])
    ax.set_ylim(0, L_total[1])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    
    colors = np.empty(Num_Particules_end, dtype = object)
    colors[0:Num_Particules_start] = "black"
    colors[Num_Particules_start:Num_Particules_end] = "red"
    #num particules - start with empty data
    init_pos = XY_master_saved[0, :, :].copy()
    s = radius_to_s(ax, Radius_molecule)
    scat = ax.scatter(init_pos[:,0], init_pos[:,1], s=s, c=colors)
    
    #define the title before
    title = ax.set_title("Particle animation")
    
    # --- choose video fps + how many sim steps per video frame ---
    fps = 30
    stride = max(1, int((1 / fps) / dt))   # sim steps per rendered frame
    frames = range(0, Nt, stride)

    #realistic aspect ratio
    ax.set_aspect("equal", adjustable="box")
    
    #vertical lines
    for k in range(1, Px):
        ax.axvline(x = (k * Lx) / Px , color = "red", linestyle = "--", linewidth = 1)
    #horizontal lines
    for k in range(1, Py):
        ax.axhline(y = (k * Ly) / Py , color = "green", linestyle = "--", linewidth = 1)
    # labels inside each tile
    for cx in range(Px):
        for cy in range(Py):
            r = cart.Get_cart_rank((cx, cy))      # rank owning that tile
            x_center = (cx + 0.5) * Lx / Px
            y_center = (cy + 0.5) * Ly / Py

            ax.text(
                x_center, y_center,
                f" {r}\n",
                ha="center", va="center",
                color="black", fontsize=9, alpha=0.9
            )

    def update(frame):  #change to have n particules 
        positions = XY_master_saved[frame, :, :] 
        # set_offsets expects an array of shape (n_points, 2)
        scat.set_offsets(positions)
        current_time = frame * dt
        ax.set_title(f"Particle animation (t = {current_time:.3f} s)")

    # --- Create animation ---  #change to have n particules 
    ani = FuncAnimation( fig, update, frames = frames, interval=1000/fps, blit = False)
    if save_gif_animation:
        writer = FFMpegWriter(fps=fps,codec="libx264",bitrate=1800)
        ani.save("particle_animation.mp4", writer=writer)
        
    

    plt.close(fig)#not showing but saving
    #time print
    t2 = time.perf_counter()
    print("For ", size, ", the runtime before the mp4 is: ", t1 - t0)
    print("For ", size, ", the runtime with the mp4 is : ", t2 - t0)
       
MPI.Finalize # stop the parallelization




