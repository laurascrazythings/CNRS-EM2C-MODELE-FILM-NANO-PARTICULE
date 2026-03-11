#funcitons needed to use the code
import numpy as np
from scipy.spatial import cKDTree #spatial tree to do a broad search
#functions to detect and solve collisions
#1 the first function is the broad phase
def broad_detect(XY, d):

    """ Goal : Detect potential collision to prevent from testing every single pair using a KD-Tree for now
        How: Returning the pairs of particles that are within a distance d of each other, meaning that they could potentially 
        collide in the next dt, and then we will test those pairs in the narrow phase to see if they actually collide or not. 
        I am using a KD-Tree for this, which is a data structure that allows for efficient spatial queries. I am also masking 
        the particles that are at position 0,0 to prevent them from being detected as colliding with each other. This is because 
        I am using the position 0,0 to represent particles that are not yet "sent" into the simulation, and I don't want them to 
        be detected as colliding with each other. I am also returning the indices of the particles in the original array, so that 
        I can easily update their positions and velocities later on.
    Args:
        XY (array): Array of particle XY positions.
        d (float): superior to the Highest distance in one dt 

    Returns:
        Array: Pairs of indices representing potential colliding particles. """   
    mask = ~np.all(XY == 0.0, axis=1)   #mask the 0.0 positioned particles
    idx_valid = np.flatnonzero(mask) #keeps the indexes
    XY_non_zero = XY[mask] #position of the real particles
    tree = cKDTree(XY_non_zero)  # Build a KD-Tree from particle positions
    pairs = np.array(list(tree.query_pairs(d)),dtype = int) # Find all pairs of particles within distance d
    return idx_valid[pairs]

#2 the second function is the narrow phase
def narrow_detect(Particle_test_pair, dt_left, Vp_stack, XY_stack, Radius_molecule, Num_Particules_end):
    """
    Goal: Test the interaction of pairs of particles that are close to each other to see if they actually 
    collide or not, and if they do, return the time of collision. I am using the equations of motion to 
    calculate the time of collision, and I am also taking into account the radius of the particles to 
    determine if they collide or not. I am also using the relative velocity and position of the particles 
    to calculate the time of collision, and I am also checking if the particles are moving towards each other 
    or not to determine if they will collide or not. If they are moving away from each other, then they will 
    not collide, even if they are close to each other. If they are moving towards each other, then I will 
    calculate the time of collision and return it, along with the pair of particles that collide. 
    
    Args:
        Particle_test_pair : array of the pair of particles collding 
        dt_left : float : dt left to the next dt
        Vp_stack : array of size 9* Number of particles and 2 dimensions: velocity of the particle
        XY_proc : array of size 9* Number of particles and 2 dimensions: position of the particle
        Radius_molecule : float : radius of the particles
        Num_Particules_end : int : number of particles in the simulation
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
def update_particles_collision(XY_stack, Vp_stack, Spin_stack, Colliding_pair, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
    """
    Goal:update the particle velocity and its position at contact - collision case
    Args:
        XY_stack: array of size 9*Number of particles and 2 dimensions: position of the particle
        Vp_stack: array of size 9*Number of particles and 2 dimensions: velocity of the particle
        Spin_stack: array of size 9*Number of particles and 2 dimensions: spin of the particle
        Colliding_pair: Array of 2 particles, of 2 dimensions
        t_collision: float: time of collision
        Added_par_stack: array of size Number of particles and 1 dimensions: to know if the particle is visible
        Radius_molecule: float: radius of the particles
        Colliding_pair: Array of 2 particles, of 2 dimensions
        t_collision: float: time of collision
        Mass_stack: array of size 9* Number of particles and 2 dimensions: mass of the particle
        Num_Particules_end: int: number of particles in the simulation
        Aggregate_set: list of the particles in the same aggregate
        Cg_stack: array of size 9* Number of particles and 2 dimensions: center of gravity of the particle
        Attributes: array of size,number of attributes, size 9, Number of particles, 2 dimensions : where all the 
        data of 1 particle is stored
        All the Index_par_..._set: sets of the indices of the particles in the local and ghost zones, to know which ones to update

        
    Returns:
        XY_stack: array of size 9*Number of particles and 2 dimensions: position of the particle updated
        Vp_stack: array of size 9* Number of particles and 2 dimensions: velocity of the particle updated
        Cg_stack: array of size 9* Number of particles and 2 dimensions: center of gravity of the particle updated
        Aggregate_set : list of the particles in the same aggregate updated
        Spin_stack: array of size 9* Number of particles and 2 dimensions: spin of the particle updated
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
    Cg_stack = Cg_stack + Vp_stack * t_collision * Added_par_stack 
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
                
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set, Spin_stack

def update_particles_aggregation(XY_stack, Vp_stack, Spin_stack, Colliding_pair, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
    """
    Goal:update the particle velocity and its position at contact - aggregation case 
    Args:
        XY_stack: array of size 9*Number of particles and 2 dimensions: position of the particle
        Vp_stack: array of size 9*Number of particles and 2 dimensions: velocity of the particle
        Spin_stack: array of size 9*Number of particles and 2 dimensions: spin of the particle
        Colliding_pair: Array of 2 particles, of 2 dimensions
        t_collision: float: time of collision
        Added_par_stack: array of size Number of particles and 1 dimensions: to know if the particle is visible
        Radius_molecule: float: radius of the particles
        Colliding_pair: Array of 2 particles, of 2 dimensions
        t_collision: float: time of collision
        Mass_stack: array of size 9* Number of particles and 2 dimensions: mass of the particle
        Num_Particules_end: int: number of particles in the simulation
        Aggregate_set: list of the particles in the same aggregate
        Cg_stack: array of size 9* Number of particles and 2 dimensions: center of gravity of the particle
        Attributes: array of size,number of attributes, size 9, Number of particles, 2 dimensions : where all the 
        data of 1 particle is stored
        All the Index_par_..._set: sets of the indices of the particles in the local and ghost zones, to know which ones to update

        
    Returns:
        XY_stack: array of size 9*Number of particles and 2 dimensions: position of the particle updated
        Vp_stack: array of size 9* Number of particles and 2 dimensions: velocity of the particle updated
        Cg_stack: array of size 9* Number of particles and 2 dimensions: center of gravity of the particle updated
        Aggregate_set : list of the particles in the same aggregate updated
        Spin_stack: array of size 9* Number of particles and 2 dimensions: spin of the particle updated
    """
    # The need for relative index comes from the fact that to have smooth collisions, I have stacked all the values of the particles into one array; to ensure I am pulling the right radius values i have to divide the index of the stack by the length of  normal array
    relative_index = np.zeros(2, dtype = int)
    relative_index[0] = Colliding_pair[0] % (Num_Particules_end )
    relative_index[1] = Colliding_pair[1] % (Num_Particules_end )
    i, j = relative_index
    
    Cgi = Cg_stack[Colliding_pair[0], :] #center of gravity of the first aggregate
    Cgj = Cg_stack[Colliding_pair[1], :] #center of gravity of the second aggregate
        
    mass_i = Mass_stack[Colliding_pair[0]]
    mass_j = Mass_stack[Colliding_pair[1]]
    
    cr = ((((2 * Radius_molecule) *10**(-6))/0.15) * (-0.88)) + 0.78 #coefficient of restitution
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
    
    #spin
    #distance from center of gravity of 
    
    for k in merge:
        Aggregate_set[k] = merge.copy()
        zone = zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
        if zone != None:
            Cg_stack[zone * Num_Particules_end + k, :] = cg_update.copy()
            if Attributes[5, zone, k, 0] == 0:
                Vp_stack[zone * Num_Particules_end + k, :] = V_new.copy() # average the velocity after impact? missing the e?
            Mass_stack[zone * Num_Particules_end + k, :] = Mass_new.copy()
   
    return XY_stack, Vp_stack, Cg_stack, Aggregate_set, Spin_stack

def zone_index(k, Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
               Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set):
    """
    from index of particle aggregating returns the zone the particle is in to allow for updating the velocity of the aggregate
    
    Args:
    k: int : index of the particle in the local array
    Index_par_..._set: sets of the indices of the particles in the local and ghost zones, to know which ones to update
    Returns:
    int: zone of the particle, from 0 to 8
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

def bounced(Index, attributes_local, aggregate_set, Local_wall, wall, Radius_molecule):
    """
    Goal: update the particle velocity and its position at contact when the particle is touching a wall
    Args:
    Index: int : index of the particle in the local array
    attributes_local: array of size,number of attributes, Number of particles, 2
    Local_wall
    Wall: string : "up", "down", "right" or "left"
    Radius_molecule: float : radius of the particles
    Returns:
    attributes_local: array of size,number of attributes, Number of particles, 2 :

    """
    #position change
    difference = 0
    if wall == "up" or wall == "down":
        axis = 1
    elif wall =="right" or wall == "left":
        axis = 0

    if wall == "up" or wall == "right":  
        difference = attributes_local[0, Index, axis] + Radius_molecule - Local_wall
    elif wall == "down" or wall == "left":
        difference = (attributes_local[0, Index, axis] - Radius_molecule - Local_wall)
        
    for k in aggregate_set:
        if (attributes_local[1,k,axis] < 0.0 and wall == "down" or wall == "left") or (attributes_local[1,k,axis] > 0.0 and wall == "up" or wall == "right"): #ensures that particules can come in if we want to add them
            #change all the positions
            attributes_local[0, k, axis] = attributes_local[0, k, axis] - difference  #set the position to the changed position after impact
            #change the velocities
            attributes_local[1, k, axis] = - attributes_local[1, k, axis]  #reverse the speed in the x direction
            #change the center of gravity
            attributes_local[2, k, axis] = attributes_local[2, k, axis] - difference #reverse the part going too far behind the limit
        
    return attributes_local
    
def adhered(Index, attributes_local, aggregate_set, Local_wall, wall, Radius_molecule):
    """
    Goal: update the particle velocity and its position at contact when the particle is touching an adhering wall
    Args:
    Index: int : index of the particle in the local array
    attributes_local: array of size,number of attributes, Number of particles, 2
    Local_wall
    Wall: string : "up", "down", "right" or "left"
    Radius_molecule: float : radius of the particles
    Returns:
    attributes_local: array of size,number of attributes, Number of particles, 2 :

    """
    #position change
    if wall == "up" or wall == "down":
        axis = 1
    elif wall =="right" or wall == "left":
        axis = 0
    if wall == "up" or "right":  
        difference = attributes_local[0, Index, axis] + Radius_molecule - Local_wall
    elif wall == "down" or "left":
        difference = (attributes_local[0, Index, axis] - Radius_molecule - Local_wall)
    
    for k in aggregate_set:
        #change all the positions
        attributes_local[0, k, axis] = attributes_local[0, k, axis] - difference   #set the position to the changed position after impact
        #change the velocities
        attributes_local[1, k, :] = [0, 0]  #reverse the speed in the x direction
        #change the center of gravity
        attributes_local[2, k, axis] = attributes_local[2, k, axis] - difference #reverse the part going too far behind the limit
        attributes_local[5, k, :] = 1
    
    
    return attributes_local