#funcitons needed to use the code
import numpy as np
from scipy.spatial import cKDTree #spatial tree to do a broad search
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
def update_particles_collision(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
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
        
    mass_i = Mass_stack[Colliding_pair[0]]
    mass_j = Mass_stack[Colliding_pair[1]]

    #update the position of the colliding particles
    XY_stack = XY_stack + Vp_stack * t_collision * Added_par_stack #update all of the particles and then later change the one which collided
    XY_stack[Colliding_pair[0] , :] = XY_stack[Colliding_pair[0] , :].copy() + u_i * t_collision
    XY_stack[Colliding_pair[1] , :] = XY_stack[Colliding_pair[1] , :].copy() + u_j * t_collision
    #Update the velocities of the particles
    distance_particles = XY_stack[Colliding_pair[0] , :] - XY_stack[Colliding_pair[1] , :] #distance vector of the particles
    dv = u_i - u_j #relative speed
    projected_distance = distance_particles @ distance_particles #||C1 - C2 ||^2
    if projected_distance > 1e-12:
        for k in Aggregate_set[i]: #update the velocities of all the particles in the aggregate
            zone = zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
            if zone != None and Attributes[5, zone, k, 0] == 0: 
                Vp_stack[zone * Num_Particules_end + k, :] = (cr * mass_j * (u_j - u_i) + mass_i * u_i + mass_j * u_j)/ (mass_i + mass_j) # average the velocity after impact? missing the e?
        for k in Aggregate_set[j]: #update the velocities of all the particles in the aggregate
            zone = zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
                              Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
            if zone != None and Attributes[5, zone, k, 0] == 0: 
                Vp_stack[zone * Num_Particules_end + k, :] = (cr * mass_i * (u_i - u_j) + mass_i * u_i + mass_j * u_j)/ (mass_i + mass_j)
                
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set,

def update_particles_aggregation(XY_stack, Vp_stack, Colliding_pair, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
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
        
    mass_i = Mass_stack[Colliding_pair[0]]
    mass_j = Mass_stack[Colliding_pair[1]]
    
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    cr = ((((2 * Radius_molecule) *10**(-6))/0.15) * (-0.88)) + 0.78
    u_i = Vp_stack[Colliding_pair[0], :].copy()
    u_j = Vp_stack[Colliding_pair[1], :].copy()
    
    #update the position of the colliding particles
    XY_stack = XY_stack+ Vp_stack * t_collision * Added_par_stack #update all of the particles and then later change the one which collided
    XY_stack[Colliding_pair[0] , :] = XY_stack[Colliding_pair[0] , :].copy() + u_i * t_collision
    XY_stack[Colliding_pair[1] , :] = XY_stack[Colliding_pair[1] , :].copy() + u_j * t_collision
    
    #they should now be glued togther?
    #updates
    merge = set(Aggregate_set[i]) #create a set that we populate and then use to replace the old set. easier
    merge.update(Aggregate_set[j])
    merge.update([i, j])
    Mass_new = Mass_stack[Colliding_pair[0]] + Mass_stack[Colliding_pair[1]]
    cg_update = (Cgi * mass_i + Cgj * mass_j) / (mass_i + mass_j)
    V_new = (u_i * mass_i + u_j * mass_j) / (mass_i + mass_j)
    
    for k in merge:
        Aggregate_set[k] = merge.copy()
        zone = zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
        if zone != None:
            Cg_stack[zone * Num_Particules_end + k, :] = cg_update.copy()
            if Attributes[5, zone, k, 0] == 0:
                Vp_stack[zone * Num_Particules_end + k, :] = V_new.copy() # average the velocity after impact? missing the e?
            Mass_stack[zone * Num_Particules_end + k, :] = Mass_new.copy()
   
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set

def zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
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
    elif k in Index_par_ghost_up_left_set:
        zone = 7
    else :
        zone = None
    return zone

def bounced(Index, attributes_local, aggregate_set, Local_wall, wall):
    """
    Docstring for bounce
    
    :param attributes_particle: Description
    :param aggregate_set_particle: Description
    """
    #position change
    if wall == "up" or wall == "down":
        axis = 1
    elif wall =="right" or wall == "left":
        axis = 0
    difference = attributes_local[0, Index, axis] - Local_wall
    for k in aggregate_set:
        #change all the positions
        attributes_local[0, k, axis] = attributes_local[0, k, axis] - difference  #set the position to the changed position after impact
        #change the velocities
        attributes_local[1, k, axis] = - attributes_local[1, k, axis]  #reverse the speed in the x direction
        #change the center of gravity
        attributes_local[2, k, axis] = attributes_local[2, k, axis] - difference #reverse the part going too far behind the limit
    
    return attributes_local
    
def adhered(Index, attributes_local, aggregate_set, Local_wall, wall):
    """
    Docstring for adhere
    
    :param Index: Description
    :param attributes_local: Description
    :param aggregate_set: Description
    :param local_wall: Description
    :param wall: Description
    """
    #position change
    if wall == "up" or wall == "down":
        axis = 1
    elif wall =="right" or wall == "left":
        axis = 0
    difference = attributes_local[0, Index, axis] - Local_wall
    
    for k in aggregate_set:
        #change all the positions
        attributes_local[0, k, axis] = attributes_local[0, k, axis] - difference  #set the position to the changed position after impact
        #change the velocities
        attributes_local[1, k, :] = [0, 0]  #reverse the speed in the x direction
        #change the center of gravity
        attributes_local[2, k, axis] = attributes_local[2, k, axis] - difference #reverse the part going too far behind the limit
        attributes_local[5, k, :] = 1
    
    
    return attributes_local