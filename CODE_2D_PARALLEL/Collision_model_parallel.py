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
from particle_interactions import update_particles_aggregation, update_particles_collision, broad_detect, narrow_detect, bounced, adhered

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
#periodic by nature but I added walls on top, I am keeping the mpi build periodic even though I am not always using it like that
up_right = cart.Get_cart_rank([coordinates_x + 1, coordinates_y + 1])
down_right = cart.Get_cart_rank([coordinates_x + 1, coordinates_y - 1])
down_left = cart.Get_cart_rank([coordinates_x - 1 , coordinates_y - 1])
up_left = cart.Get_cart_rank([coordinates_x - 1, coordinates_y + 1])

#initialize the plot
plt.clf() 

#USER INIT
#time - TO SET
T = 40# seconds to change - HERE
T_add_particles = 2 #time for which I add particles for - HERE
dt = 0.05 #delta t 
#mp4 animation
save_gif_animation = True # save animation as a mp4? - To SET 
#Mesh - TO SET
L_total = np.array([100, 100]) #Total Size in microm - HERE
#Particles - TO SET
position = 0 #0 for auto and 1 for manual choice
Num_Particules = 2000#particles to start - HERE
Num_Particules_dt= 0 #particls added per second - HERE
#TiO2 properties - rutile for now
A_h = 6*10**(-20) #hamaker constant for rutile Tio2
Radius_molecule = 0.02 #radius of the particule in micrometer 
density = 4500000 #g/m3
Molar_mass = 79.9 # we can put the molecular mass in g/mol because we are only using this value for ratio calculation
#Velocity of particles
Highest_velocity = 1 #velocity of the particle
Lowest_velocity = 0.01
Lowest_x_velocity = -1
Highest_x_velocity = 1
#Brownian
tau = 1.0 # relaxation time for Ornstein Uhlenbeck, chosen as 1 for now 
B = 0.5 # noise intensity
#Gas velocity
U_g = np.array([0, 1]) # mean gas velocity
#walls : is there a wall, #0 for right, 1 for left, 2 for up, 3 for bottom, 4 for none, if none periodical condition
wall = set(); 
wall.add(2)
wall.add(3)
#adhering ?: is the wall adhering?  #0 for right, 1 for left, 2 for up, 3 for bottom, 4 for none
adhere = set() 
adhere.add(2)

# init var - DO NOT TOUCH
Nt = int(T/ dt) #num of Iterations
Lx, Ly = L_total # dimensions of the domain
Px, Py = dimensions_proc #neded to set the lines later 
#total number of particles by the end to create the right arrays and not resize(costly) - DO NOT TOUCH
Num_Particules_start = Num_Particules
Num_Particules_end = int(Num_Particules +  Num_Particules_dt * (T_add_particles /dt))

#define the charatcters of the particles
Volume = 4/3 * np.pi * (Radius_molecule*10**(-6))**3
Mass_one_particle = density * Volume * 10**(12) #picograms

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
    Added_par = np.zeros((Num_Particules_end)) 
    Added_par[:Num_Particules] = 1 #1 equals true  
    
    #create the random xi values for the brownian now so we an broadcast them and have a common value for them
    xi = np.random.randn(Nt, Num_Particules_end, 2)
else: 
    XY_start = None
    Vp = None
    Added_par = None
    Buffer_zone_width = None
    xi = None
    
#broadcast the values to the different procs
XY_start = cart.bcast(XY_start, root = 1)
Vp = cart.bcast(Vp, root = 1)
Added_par = cart.bcast(Added_par, root = 1)
Buffer_zone_width = cart.bcast(Buffer_zone_width, root = 1)
xi = cart.bcast(xi, root = 1)
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

# Particule attributes - set to zeros the attributes
Attributes = np.zeros((6, 9, Num_Particules_end, 2)) #creating the different attributes
# Attributes[0, :,:, :] : position; 1: velocity; 2: Cg_proc; 3: Mass; 4: Display on or off; 5: Attached to the wall 

##[:,0] : right; [:,1]: left; [:,2]: up; [:,3]: down [:,4] : up right; [:,5]: down right; [:,6]: down left; [:,7]: up left; [:. 8]: local
Aggregate_set = [{p} for p in range(Num_Particules_end)]
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
        Attributes[0, 8, p, :] = XY_start[p, :] #position
        Attributes[1, 8, p, :] = Vp[p, :] #velocity
        Attributes[2, 8, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 8, p, :] = [Mass_one_particle, Mass_one_particle] #mass, the 0 is to not get errors
        Attributes[4, 8, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 8, p, :] = [0, 0] #attached to a wall, the first 0 is for false and the second to fill the space to not get errors
    elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up: #particule in the left ghost
        Index_par_ghost_left.append(p)
        Index_par_ghost_left_set.add(p)
        Attributes[0, 1, p, :] = XY_start[p, :] #position
        Attributes[1, 1, p, :] = Vp[p, :] #velocity
        Attributes[2, 1, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 1, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 1, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 1, p, :] = [0, 0] #attached to a wall 
    elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y < Local_up: #particule in the right ghost
        Index_par_ghost_right.append(p)
        Index_par_ghost_right_set.add(p)
        Attributes[0, 0, p, :] = XY_start[p, :] #position
        Attributes[1, 0, p, :] = Vp[p, :] #velocity
        Attributes[2, 0, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 0, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 0, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 0, p, :] = [0, 0] #attached to a wall 
    elif Local_left <= Position_x < Local_right and Local_up <= Position_y <= Local_ghost_up: #up ghost 
        Index_par_ghost_up.append(p)
        Index_par_ghost_up_set.add(p)
        Attributes[0, 2, p, :] = XY_start[p, :] #position
        Attributes[1, 2, p, :] = Vp[p, :] #velocity
        Attributes[2, 2, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 2, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 2, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 2, p, :] = [0, 0] #attached to a wall  
    elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down: #down ghost 
        Index_par_ghost_down.append(p)
        Index_par_ghost_down_set.add(p)
        Attributes[0, 3, p, :] = XY_start[p, :] #position
        Attributes[1, 3, p, :] = Vp[p, :] #velocity
        Attributes[2, 3, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 3, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 3, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 3, p, :] = [0, 0] #attached to a wall 
    elif Local_right <= Position_x <= Local_ghost_right and Local_up <= Position_y <= Local_ghost_up : #corners up right ghost
        Index_par_ghost_up_right.append(p)
        Index_par_ghost_up_right_set.add(p)
        Attributes[0, 4, p, :] = XY_start[p, :] #position
        Attributes[1, 4, p, :] = Vp[p, :] #velocity
        Attributes[2, 4, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 4, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 4, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 4, p, :] = [0, 0] #attached to a wall  
    elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down : #corners down right ghost
        Index_par_ghost_down_right.append(p)
        Index_par_ghost_down_right_set.add(p)
        Attributes[0, 5, p, :] = XY_start[p, :] #position
        Attributes[1, 5, p, :] = Vp[p, :] #velocity
        Attributes[2, 5, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 5, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 5, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 5, p, :] = [0, 0] #attached to a wall 
    elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down : #corners down left ghost
        Index_par_ghost_down_left.append(p)
        Index_par_ghost_down_left_set.add(p)
        Attributes[0, 6, p, :] = XY_start[p, :] #position
        Attributes[1, 6, p, :] = Vp[p, :] #velocity
        Attributes[2, 6, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 6, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 6, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 6, p, :] = [0, 0] #attached to a wall  
    elif Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up : #corners up left ghost
        Index_par_ghost_up_left.append(p)
        Index_par_ghost_up_left_set.add(p)
        Attributes[0, 7, p, :] = XY_start[p, :] #position
        Attributes[1, 7, p, :] = Vp[p, :] #velocity
        Attributes[2, 7, p, :] = XY_start[p, :] #Center of gravity is equal to the position unless in an aggregate
        Attributes[3, 7, p, :] = [Mass_one_particle, Mass_one_particle] #mass
        Attributes[4, 7, p, :] = [Added_par[p], Added_par[p]] # display particules, 0 to not get errors
        Attributes[5, 7, p, :] = [0, 0] #attached to a wall  
    #because of the periodical condition, the ghosts of the borders is actually on the other side, it can cause some distubances in the t =0 because the particle dont fall into any categories    
         

#initiate the saving of information
#I wonder if I should print the particles in the ghost areas as well in different colors... For now, only the local xy will be printed
XY_master_saved = np.zeros((Nt, Num_Particules_end, 2)) #use at the end when merging the info
XY_local_saved = np.zeros((Nt, Num_Particules_end, 2)) #saving the positions of every particle of each processor in the time loop, will merge the info at the end
XY_local_saved[0,:,:] = Attributes[0, 8, :, :].copy()

# Boolean value that will keep track of if i have already sent a particule to another proc so 
# that it doesnt resend it, will be reset to false when the particule leaves the proc t enable rollback
Local_sent = np.zeros((Num_Particules_end, 8), dtype = bool) #[:,0] : right; [:,1]: left; [:,2]: up; [:,3]: down [:,4] : up right; [:,5]: down right; [:,6]: down left; [:,7]: up left


#time loop to update the map
for t in range(1, Nt + 1):
    #modify xi depending on the aggregate to have the same value for the aggregate lists
    for i in range(Num_Particules_end):
        if len(Aggregate_set[i]) > 1: #1 2 3 4 5 
            Avg_Brownian = np.zeros(2) #initiate the avg 
            for k in Aggregate_set[i]: #go through all the 
                Avg_Brownian = Avg_Brownian + xi[t - 1, k, :]
            Avg_Brownian = Avg_Brownian / len(Aggregate_set[i])    
            xi[t - 1, i, :] = Avg_Brownian
    # #update the velocities to match a brownian motion, we chose to go with the ornstein uhlenbeck implementation on speed
    non_zero_gaussian = ~np.all(Attributes[1,:,:,:] == 0.0)
    index_non_zero_gaussian = np.flatnonzero(non_zero_gaussian)
    # for i in range(len(index_non_zero_gaussian)):
    #     idx = index_non_zero_gaussian[i]
    #     first_value = min(Aggregate_set[idx])
    #     xi[t - 1, idx, :] = xi[t - 1, first_value, :]
    Attributes[1,:, index_non_zero_gaussian, :] = Attributes[1, :, index_non_zero_gaussian, :] - (Attributes[1, :, index_non_zero_gaussian, :] - U_g) * (dt/tau) + np.sqrt(B * dt) * xi[t-1, index_non_zero_gaussian, :][:, None, :]
    # #if Vp != 0.0 :  (doesnt work with aggregates bcause the velocities are dependant)
    #     Vp_proc[8,:,:] = Vp_proc[8,:,:] - ((Vp_proc[8,:,:] - U_g) * (dt/tau) + np.sqrt(B * dt) * xi)
    if rank == 0 and ((t * dt) % 2 ) == 0 :
        print(t*dt)
      #particles are getting added, add a certain number of particles, I have decided to do a boolean that changes value depending on the rate at which we add particle. The particle are "invisible" or at least wont move position until they are able to
    if t*dt <= T_add_particles:
        Attributes[4, :, Num_Particules : (Num_Particules + Num_Particules_dt), :] = 1
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
        XY_stack = Attributes[0,:,:,:].reshape(-1, 2)# stack the position values to allow for proimity checks
        Vp_stack = Attributes[1,:,:,:].reshape(-1, 2)
        Cg_stack = Attributes[2,:,:,:].reshape(-1, 2)
        Mass_stack = Attributes[3,:,:,:].reshape(-1, 2)
        Added_par_stack = Attributes[4, :, :, :].reshape(-1, 2)
        
        
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
                    if link > 0.5 :
                        XY_stack, Vp_stack, Cg_stack, Aggregate_set = update_particles_collision(XY_stack, Vp_stack, First_collision, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes,
                                                                                                 Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
                                                                                                 Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
                    else:
                        XY_stack, Vp_stack, Cg_stack, Aggregate_set = update_particles_aggregation(XY_stack, Vp_stack, First_collision, t_collision, Added_par_stack, Radius_molecule, Mass_stack, Num_Particules_end, Aggregate_set, Cg_stack, Attributes,
                                                                                                   Index_par_local_set, Index_par_ghost_right_set, Index_par_ghost_left_set, Index_par_ghost_up_set, Index_par_ghost_down_set, 
                                                                                                   Index_par_ghost_up_right_set, Index_par_ghost_down_right_set, Index_par_ghost_down_left_set, Index_par_ghost_up_left_set)
                    dt_left = dt_left - t_collision
                else:
                    XY_stack = XY_stack + Vp_stack * dt_left * Added_par_stack
                    dt_left = 0
            else:
                XY_stack = XY_stack + Vp_stack * dt_left * Added_par_stack             
                dt_left = 0 
              
        Attributes[0,:,:,:] = XY_stack.reshape(9, Num_Particules_end, 2)
        Attributes[1,:,:,:] = Vp_stack.reshape(9, Num_Particules_end, 2)
        Attributes[2,:,:,:] = Cg_stack.reshape(9, Num_Particules_end, 2)
        Attributes[3,:,:,:] = Mass_stack.reshape(9, Num_Particules_end, 2)
        #now we have dealt with the transition from local to the ghosts, 
        # we have to deal with inter ghost and leaving interactions, for each ghost, there are 4 interactions possible.
        #right ghost interactions            
        for par_right in reversed(range(len(Index_par_ghost_right))): #in the ghost right area
            Index = Index_par_ghost_right[par_right] #index of the particule in the ghost right area to use for the xy vp which are indexed following the grand scheme to obliviate confusion
            Position_x = Attributes[0, 0, Index, 0]
            Position_y = Attributes[0, 0, Index, 1]
            #4 cases for now forward or backward or up or down
            #forward right to leave
            if Position_x > Local_ghost_right:
                #particule is leaving the ghost right area to nowhere as it is already sent to the next one
                Attributes[:, 0, Index, :] = [0, 0]
                Index_par_ghost_right.pop(par_right) #remove the particule from the count
                Index_par_ghost_right_set.discard(Index)
            #roll back right to local
            elif Position_x < Local_right and Local_down <= Position_y < Local_up:
                #the particule enters the local area and leaves the ghost right area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 0, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 0, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_right.pop(par_right) #Index of the new particule added
                Index_par_ghost_right_set.discard(Index)
            #up to the ghost up-right 
            elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_up >= Position_y >= Local_up: 
                if Local_up != L_total[1] or 2 not in wall:
                    #the particle enters the up right ghost
                    Index_par_ghost_up_right.append(Index) #add the particle to the list
                    Index_par_ghost_up_right_set.add(Index)
                    Attributes[:, 4, Index, :] = Attributes[:, 0, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 0, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_right.pop(par_right) #Index of the new particule added
                    Index_par_local_set.discard(Index)
                else :
                    if 2 not in adhere:
                        Attributes[:, 0, :, :] = bounced(Index, Attributes[:, 0, :, :], Aggregate_set[Index], Local_up, "up") #this function changes the velocities, center of gravity and position to propagate the 
                    else: 
                        #if the wall is sticky then the whole aggregate has to loose velocity
                        Attributes[:, 0, :, :] = adhered(Index, Attributes[:, 0, :, :], Aggregate_set[Index], Local_up, "up")
                        #for all of the aggregate  velocity = 0, 0, if collision the velocity remains 0?
            #down to the down ghost - down right
            elif Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down: 
                #the particle enters the down right ghost
                Index_par_ghost_down_right.append(Index) #add the particle to the list
                Index_par_ghost_down_right_set.add(Index)
                Attributes[:, 5, Index, :] = Attributes[:, 0, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 0, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_right.pop(par_right) #Index of the new particule added
                Index_par_local_set.discard(Index)
        
        #left ghost interactions    
        for par_left in reversed(range(len(Index_par_ghost_left))):
            Index = Index_par_ghost_left[par_left] #index of the particule in the ghost right area
            Position_x = Attributes[0, 1, Index, 0]
            Position_y = Attributes[0, 1, Index, 1]
            #4cases in 2d
            #forward left to local 
            if Position_x >= Local_left and Local_down <= Position_y < Local_up :
                #the particule enters the local area and leaves the ghost left area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 1, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 1, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_left.pop(par_left) #Index of the new particule added
                Index_par_ghost_left_set.discard(Index)
            #rollback left to leave
            elif Position_x < Local_ghost_left : 
                #particule is leaving the ghost right area 
                Attributes[:, 1, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_left.pop(par_left) #remove the particule from the count
                Index_par_ghost_left_set.discard(Index)
            #down to down left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down:
                if Local_down != 0 or 3 not in wall:
                    #particle enters the down left area
                    Index_par_ghost_down_left.append(Index) #add the index to the list
                    Index_par_ghost_down_left_set.add(Index)
                    Attributes[:, 6, Index, :] = Attributes[:, 1, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 1, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_left.pop(par_left) #Index of the new particule added
                    Index_par_ghost_left_set.discard(Index)
                else:
                    Attributes[:, 1, :, :] = bounced(Index, Attributes[:, 1, :, :], Aggregate_set[Index], Local_down, "down") #this function changes the velocities, center of gravity and position to propagate the 
                
            #up to up left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_up >= Position_y >= Local_up:
                if Local_up != L_total[1] or 2 not in wall:
                    #particle enters the up left area
                    Index_par_ghost_up_left.append(Index) #add the index to the list
                    Index_par_ghost_up_left_set.add(Index)
                    Attributes[:, 7, Index, :] = Attributes[:, 1, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 1, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_left.pop(par_left) #Index of the new particle added
                    Index_par_ghost_left_set.discard(Index)
                else :
                    if 2 not in adhere:
                        Attributes[:, 1, :, :] = bounced(Index, Attributes[:, 1, :, :], Aggregate_set[Index], Local_up, "up") #this function changes the velocities, center of gravity and position to propagate the 
                    else: 
                        #if the wall is sticky then the whole aggregate has to loose velocity
                        Attributes[:, 1, :, :] = adhered(Index, Attributes[:, 1, :, :], Aggregate_set[Index], Local_up, "up")
                        #for all of the aggregate  velocity = 0, 0, if collision the velocity remains 0?
                
        
        #up ghost interactions
        for par_up in reversed(range(len(Index_par_ghost_up))):
            Index = Index_par_ghost_up[par_up] #index of the particule in the ghost up area
            Position_x = Attributes[0, 2, Index, 0]
            Position_y = Attributes[0, 2, Index, 1]
            #4 cases in 2d
            #forward - up to up right
            if Local_right <= Position_x <= Local_ghost_right and Local_up <= Position_y <= Local_ghost_up:
                #if it is a wall then it bounces
                if Local_right != L_total[0] or 0 not in wall:  
                    #enters the up right
                    Index_par_ghost_up_right.append(Index) #add the index to the list
                    Index_par_ghost_up_right_set.add(Index)
                    Attributes[:, 4, Index, :] = Attributes[:, 2, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 2, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_up.pop(par_up) #Index of the new particule added
                    Index_par_ghost_up_set.discard(Index)
                else:
                    Attributes[:, 2, :, :] = bounced(Index, Attributes[:, 2, :, :], Aggregate_set[Index], Local_right, "right") #this function changes the velocities, center of gravity and position to propagate the bouncing to the whole aggregate   
            #backward - up to up left
            elif Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up:
                if Local_left != 0 or 1 not in wall:
                    #enters the up left
                    Index_par_ghost_up_left.append(Index) #add the index to the list
                    Index_par_ghost_up_left_set.add(Index)
                    Attributes[:, 7, Index, :] = Attributes[:, 2, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 2, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_up.pop(par_up) #Index of the new particule added
                    Index_par_ghost_up_set.discard(Index)
                else:
                    Attributes[:, 2, :, :] = bounced(Index, Attributes[:, 2, :, :], Aggregate_set[Index], Local_left, "left") #this function changes the velocities, center of gravity and position to propagate the 
            #up - leaves
            elif Position_y > Local_ghost_up:
                #leaves the proc area
                Attributes[:, 2, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
            #down - local
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                #enters the local
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 2, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 2, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up.pop(par_up) #Index of the new particule added
                Index_par_ghost_up_set.discard(Index)
        
        #down ghost interactions
        for par_down in reversed(range(len(Index_par_ghost_down))):
            Index = Index_par_ghost_down[par_down] #index of the particule in the ghost down area
            Position_x = Attributes[0, 3, Index, 0]
            Position_y = Attributes[0, 3, Index, 1]
            #4 cases in 2d
            #forward - down to down right
            if Local_right <= Position_x <= Local_ghost_right and Local_ghost_down <= Position_y < Local_down:
                #if it is a wall then it bounces
                if Local_right != L_total[0] or 0 not in wall:  
                    #enters the down right
                    Index_par_ghost_down_right.append(Index) #add the index to the list
                    Index_par_ghost_down_right_set.add(Index)
                    Attributes[:, 5, Index, :] = Attributes[:, 3, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 3, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_down.pop(par_down) #Index of the new particule added
                    Index_par_ghost_down_set.discard(Index)
                else:
                    Attributes[:, 3, :, :] = bounced(Index, Attributes[:, 3, :, :], Aggregate_set[Index], Local_right, "right") #this function changes the velocities, center of gravity and position to propagate the bouncing to the whole aggregate   
            #backward - down to down left
            elif Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down:
                if Local_left != 0 or 1 not in wall:
                    #enters the down left
                    Index_par_ghost_down_left.append(Index) #add the index to the list
                    Index_par_ghost_down_left_set.add(Index)
                    Attributes[:, 6, Index, :] = Attributes[:, 3, Index, :].copy() #update the local position of the particule, its velocity and all
                    Attributes[:, 3, Index, :] = [0, 0] #remove the attributes to this position
                    Index_par_ghost_down.pop(par_down) #Index of the new particule added
                    Index_par_ghost_down_set.discard(Index)
                else:
                    Attributes[:, 3, :, :] = bounced(Index, Attributes[:, 3, :, :], Aggregate_set[Index], Local_left, "left") #this function changes the velocities, center of gravity and position to propagate the 
            #down - leaves
            elif Position_y < Local_ghost_down:
                #leaves the proc area
                Attributes[:, 3, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            #up - local
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                #enters the local
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 3, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 3, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down.pop(par_down) #Index of the new particule added
                Index_par_ghost_down_set.discard(Index)
            
        #up right ghost interactions
        for par_up_right in reversed(range(len(Index_par_ghost_up_right))):
            Index = Index_par_ghost_up_right[par_up_right] #index of the particule in the ghost up right area
            Position_x = Attributes[0, 4, Index, 0]
            Position_y = Attributes[0, 4, Index, 1]
            #3 cases in 2d
            #forward or up - upright to leave
            # up or right - leaves
            if Position_y > Local_ghost_up or Position_x > Local_ghost_right:
                #leaves the proc area
                Attributes[:, 4, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #down - ghost right
            elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y <= Local_up:
                #enters the ghost right
                Index_par_ghost_right.append(Index) #add the index to the list
                Index_par_ghost_right_set.add(Index)
                Attributes[:, 0, Index, :] = Attributes[:, 4, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 4, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #left - ghost up
            elif Local_left <= Position_x < Local_right and Local_up <= Position_y < Local_ghost_up:
                #enters the ghost up
                Index_par_ghost_up.append(Index) #add the index to the list
                Index_par_ghost_up_set.add(Index)
                Attributes[:, 2, Index, :] = Attributes[:, 4, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 4, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 4, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 4, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_right.pop(par_up_right) #Index of the new particule added
                Index_par_ghost_up_right_set.discard(Index)
        
        #down right ghost interaction
        for par_down_right in reversed(range(len(Index_par_ghost_down_right))):
            Index = Index_par_ghost_down_right[par_down_right] #index of the particule in the ghost down right area
            Position_x = Attributes[0, 5, Index, 0]
            Position_y = Attributes[0, 5, Index, 1]
            #3 cases in 2d
            #forward or down - down right to leave
            # down or right - leaves
            if Position_y < Local_ghost_down or Position_x > Local_ghost_right:
                #leaves the proc area
                Attributes[:, 5, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #up - ghost right
            elif Local_right <= Position_x <= Local_ghost_right and Local_down <= Position_y < Local_up:
                #enters the ghost right
                Index_par_ghost_right.append(Index) #add the index to the list
                Index_par_ghost_right_set.add(Index)
                Attributes[:, 0, Index, :] = Attributes[:, 5, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 5, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #left - ghost down
            elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down:
                #enters the ghost down
                Index_par_ghost_down.append(Index) #add the index to the list
                Index_par_ghost_down_set.add(Index)
                Attributes[:, 3, Index, :] = Attributes[:, 5, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 5, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 5, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 5, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_right.pop(par_down_right) #Index of the new particule added
                Index_par_ghost_down_right_set.discard(Index)
                
        #down left ghost interactions
        for par_down_left in reversed(range(len(Index_par_ghost_down_left))):
            Index = Index_par_ghost_down_left[par_down_left] #index of the particule in the ghost down left area
            Position_x = Attributes[0, 6, Index, 0]
            Position_y = Attributes[0, 6, Index, 1]
            #3 cases in 2d
            #backward or down - down left to leave
            # down or left - leaves
            if Position_y < Local_ghost_down or Position_x < Local_ghost_left:
                #leaves the proc area
                Attributes[:, 6, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #up - ghost left
            elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up:
                #enters the ghost left
                Index_par_ghost_left.append(Index) #add the index to the list
                Index_par_ghost_left_set.add(Index)
                Attributes[:, 1, Index, :] = Attributes[:, 6, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 6, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #right - ghost down
            elif Local_left <= Position_x < Local_right and Local_ghost_down <= Position_y < Local_down:
                #enters the ghost down
                Index_par_ghost_down.append(Index) #add the index to the list
                Index_par_ghost_down_set.add(Index)
                Attributes[:, 3, Index, :] = Attributes[:, 6, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 6, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 6, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 6, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_down_left.pop(par_down_left) #Index of the new particule added
                Index_par_ghost_down_left_set.discard(Index)
        
        #up left ghost interactions
        for par_up_left in reversed(range(len(Index_par_ghost_up_left))):
            Index = Index_par_ghost_up_left[par_up_left] #index of the particule in the ghost up left area
            Position_x = Attributes[0, 7, Index, 0]
            Position_y = Attributes[0, 7, Index, 1]
            #3 cases in 2d
            #backward or down - down left to leave
            # up or left - leaves
            if Position_y > Local_ghost_up or Position_x < Local_ghost_left:
                #leaves the proc area
                Attributes[:, 7, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #down - ghost left
            elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up:
                #enters the ghost left
                Index_par_ghost_left.append(Index) #add the index to the list
                Index_par_ghost_left_set.add(Index)
                Attributes[:, 1, Index, :] = Attributes[:, 7, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 7, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #right - ghost up
            elif Local_left <= Position_x < Local_right and Local_up <= Position_y <= Local_ghost_up:
                #enters the ghost up
                Index_par_ghost_up.append(Index) #add the index to the list
                Index_par_ghost_up_set.add(Index)
                Attributes[:, 2, Index, :] = Attributes[:, 7, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 7, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
            #local - if it just updates to be in the local - rarely but happens
            elif Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up:
                Index_par_local.append(Index)
                Index_par_local_set.add(Index)
                Attributes[:, 8, Index, :] = Attributes[:, 7, Index, :].copy() #update the local position of the particule, its velocity and all
                Attributes[:, 7, Index, :] = [0, 0] #remove the attributes to this position
                Index_par_ghost_up_left.pop(par_up_left) #Index of the new particule added
                Index_par_ghost_up_left_set.discard(Index)
        
        #update the local first if it is out of bounds 
        for par in reversed(range(len(Index_par_local))):
            Index = Index_par_local[par] #index in the xy, vp and other
            Position_x = Attributes[0, 8, Index, 0]
            Position_y = Attributes[0, 8, Index, 1]                
            #gathering the non covering cnditions to improve run time if and elifs that dont break the code basically
            #moving right -- technically but also called when moving backward
            if (Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and (Local_down + Buffer_zone_width[1]) <= Position_y < (Local_up - Buffer_zone_width[1]) and not(Local_right == L_total[0] and 0 in wall)): #xy >= ghost left du prochain #and not Local_sent[Index, 0]
                #The particule entered the left ghost zone of the right processor, 
                #we send it to the right proc if we havent yet
                Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                Local_sent[Index,0] = True
            #moving left
            elif (Local_left <= Position_x <= (Local_left + Buffer_zone_width[0]) and (Local_down + Buffer_zone_width[1]) <= Position_y < (Local_up - Buffer_zone_width[1]) and not(Local_left == 0 and 1 in wall)): 
                #we send it to the left proc if we havent yet
                Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, position, velocity
                Local_sent[Index, 1] = True
            #up
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and (Local_left + Buffer_zone_width[0]) <= Position_x < (Local_right - Buffer_zone_width[0]) and not(Local_up == L_total[1] and 2 in wall)): 
                #the particle has entered the down ghost zone of the up proc
                Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
            #down
            elif (Local_down <= Position_y <= (Local_down + Buffer_zone_width[1]) and (Local_left + Buffer_zone_width[0]) <= Position_x < (Local_right - Buffer_zone_width[0]) and not(Local_down == 0 and 3 in wall)): 
                #the particle has entered the up ghost zone of the down particle
                Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
            #up-right  
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and not(Local_right == L_total[0] and 0 in wall) and not(Local_up == L_total[1] and 2 in wall)):
                #enters the down left of the up right particle, but also the down of the up and the left of the right
                #down left of the up right
                Particle_info_up_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                Local_sent[Index, 4] = True
                #down of the up
                Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
                #left of the right
                Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, position, velocity
                Local_sent[Index, 0] = True   
            #down-right
            elif (Local_down <= Position_y < (Local_down + Buffer_zone_width[1]) and Local_right > Position_x >= (Local_right - Buffer_zone_width[0]) and not(Local_right == L_total[0] and 0 in wall) and not(Local_down == 0 and 3 in wall)): 
                #enters the up left ghost of the down right proc but also the up of the down and the left of the right
                Particle_info_down_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                Local_sent[Index, 5] = True
                #up of the down
                Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
                #left of the right
                Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, position, velocity
                Local_sent[Index, 0] = True
            #down-left
            elif (Local_down < Position_y < (Local_down + Buffer_zone_width[1]) and Local_left < Position_x <= (Local_left + Buffer_zone_width[0]) and not(Local_left == 0 and 1 in wall) and not(Local_down == 0 and 3 in wall)):
                #enters the up right ghost of the down left particle but also in the up of the down and the right of the left
                Particle_info_down_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                Local_sent[Index, 6] = True
                #up of the down
                Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 3] = True
                #right of the left
                Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                Local_sent[Index, 1] = True   
            #up left
            elif (Local_up > Position_y >= (Local_up - Buffer_zone_width[1]) and Local_left < Position_x <= (Local_left + Buffer_zone_width[0]) and not(Local_left == 0 and 1 in wall) and not(Local_up == L_total[1] and 2 in wall)):
                #enters the down right ghost of the up left particle but also down of ht ep and right of the left
                Particle_info_up_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                Local_sent[Index, 7] = True
                #down of the up 
                Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                Local_sent [Index, 2] = True
                #right of the left
                Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                Local_sent[Index, 1] = True 
             
                
            #particule is leaving the local area for a ghost position    
            #nominally in the same proc it should only be in one space so elifs work because only one condition may be true.
            #local to right ghost
            if Position_x >= Local_right and Local_up > Position_y >= Local_down:
                #if right is a wall then change the position and velocity if not then pass it on to the right ghost 
                if Local_right != L_total[0] or 0 not in wall:  
                    #the particule left the main local proc area, enters the right ghost area 
                    Index_par_ghost_right.append(Index)#add the index to the end of the list  
                    Index_par_ghost_right_set.add(Index)    
                    Attributes[:, 0, Index, :] = Attributes[:, 8, Index, :].copy() #update the local position of the particule, its velocity and all
                    if Local_sent[Index, 0] == False: #saved a particle that changed direction while being in the boundary area
                        Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index,0] = True
                    Attributes[:, 8, Index, :] = [0, 0] #remove the attributes from this position
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_right, "right") #this function changes the velocities, center of gravity and position to propagate the bouncing to the whole aggregate        
            #local to left ghost 
            elif Position_x < Local_left and Local_down <= Position_y < Local_up :
                #if left is a wall then change the position and velocity if not then pass it on to the left ghost 
                if Local_left != 0 or 1 not in wall:
                    #the particule left the main local proc area, enters the left ghost area    
                    Index_par_ghost_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_left_set.add(Index)       
                    Attributes[:, 1, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost 
                    if Local_sent[Index, 1] == False:
                        Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index, 1] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00 
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_left, "left") #this function changes the velocities, center of gravity and position to propagate the 
            #local to up 
            elif(Position_y >= Local_up and Local_left <= Position_x < Local_right):
                #if up is a wall then change the position and velocity if not then pass it on to the up ghost 
                if Local_up != L_total[1] or 2 not in wall:
                    #the particle enters the ghost up area and leaves the local
                    Index_par_ghost_up.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_set.add(Index)
                    Attributes[:, 2, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 2] == False:
                        Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index,2] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else :
                    if 2 not in adhere:
                        Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up") #this function changes the velocities, center of gravity and position to propagate the 
                    else: 
                        #if the wall is sticky then the whole aggregate has to loose velocity
                        Attributes[:, 8, :, :] = adhered(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up")
                        #for all of the aggregate  velocity = 0, 0, if collision the velocity remains 0?
            #local to down
            elif(Position_y < Local_down and Local_left <= Position_x < Local_right):
                #if down is a wall then change the position and velocity if not then pass it on to the down ghost 
                if Local_down != 0 or 3 not in wall:
                    #the particle enters the ghost down area and leaves the local
                    Index_par_ghost_down.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_set.add(Index)
                    Attributes[:, 3, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 3] == False:
                        Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index,3] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                else:
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_down, "down") #this function changes the velocities, center of gravity and position to propagate the 
            #local to up right    
            elif Position_x >= Local_right and Local_up <= Position_y <= Local_ghost_up:
                #if up and right are a wall then change the position and velocity if not then pass it on to the up right ghost 
                if (Local_up != L_total[1] or 2 not in wall) and (Local_right != L_total[0] or 0 not in wall):
                    #particle enters the ghost up right area
                    Index_par_ghost_up_right.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_right_set.add(Index)
                    Attributes[:, 4, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 4] == False:
                        Particle_info_up_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index, 4] = True
                        #down of the up
                        Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                        Local_sent [Index, 2] = True
                        #left of the right
                        Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, position, velocity
                        Local_sent[Index, 0] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_up == L_total[1] and 2 in wall): 
                    if 2 not in adhere:
                        Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up") #this function changes the velocities, center of gravity and position to propagate the 
                    else:
                        #if the wall is sticky then the whole aggregate has to loose velocity
                        Attributes[:, 8, :, :] = adhered(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up")
                        #for all of the aggregate velocity = 0, 0, if collision the velocity remains 0?
                        
                if (Local_right == L_total[0] and 0 in wall):
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_right, "right") #this function changes the velocities, center of gravity and position to propagate the 
                    
            #local to down right
            elif Position_x >= Local_right and Local_ghost_down <= Position_y < Local_down:
                #if down and right are a wall then change the position and velocity if not then pass it on to the down right ghost 
                if (Local_down != 0 or 3 not in wall) and (Local_right != L_total[0] or 0 not in wall):
                    #particle enters the ghost down right area
                    Index_par_ghost_down_right.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_right_set.add(Index)
                    Attributes[:, 5, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 5] == False:
                        #enters the up left ghost of the down right proc but also the up of the down and the left of the right
                        Particle_info_down_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                        Local_sent[Index, 5] = True
                        #up of the down
                        Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                        Local_sent [Index, 3] = True
                        #left of the right
                        Particle_info_right.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, position, velocity
                        Local_sent[Index, 0] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_down == 0 and 3 in wall): #only the right bounces
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_down, "down") #this function changes the velocities, center of gravity and position to propagate the 
                if (Local_right == L_total[0] and 0 in wall):
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_right, "right") #this function changes the velocities, center of gravity and position to propagate the 
            #local to down left
            elif (Local_ghost_left <= Position_x < Local_left and Local_ghost_down <= Position_y < Local_down):
                #if down and left are a wall then change the position and velocity if not then pass it on to the down left ghost 
                if (Local_down != 0 or 3 not in wall) and (Local_left != 0 or 1 not in wall):
                    #particle enters the down left area
                    Index_par_ghost_down_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_down_left_set.add(Index)
                    Attributes[:, 6, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 6] == False:
                        #enters the up right ghost of the down left particle but also in the up of the down and the right of the left
                        Particle_info_down_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                        Local_sent[Index, 6] = True
                        #up of the down
                        Particle_info_down.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                        Local_sent [Index, 3] = True
                        #right of the left
                        Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index, 1] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if (Local_down == 0 and 3 in wall):
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_down, "down") #this function changes the velocities, center of gravity and position to propagate the 
                if (Local_left == 0 and 1 in wall) :
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_left, "left") #this function changes the velocities, center of gravity and position to propagate the 
            #local to up left
            elif (Local_ghost_left <= Position_x < Local_left and Local_up <= Position_y <= Local_ghost_up):
                #if up and left are a wall then change the position and velocity if not then pass it on to the up left ghost 
                if (Local_up != L_total[1] or 2 not in wall) and (Local_left != 0 or 1 not in wall):
                    #particle enters the up left area
                    Index_par_ghost_up_left.append(Index)#add the index to the end of the list
                    Index_par_ghost_up_left_set.add(Index)
                    Attributes[:, 7, Index, :] = Attributes[:, 8, Index, :].copy() #associate the local attributes with the ghost
                    if Local_sent[Index, 7] == False:
                        #enters the down right ghost of the up left particle but also down of ht ep and right of the left
                        Particle_info_up_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy()))
                        Local_sent[Index, 7] = True
                        #down of the up 
                        Particle_info_up.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) #index, position, velocity
                        Local_sent [Index, 2] = True
                        #right of the left
                        Particle_info_left.append((Index, Attributes[:, 8, Index, :].copy(), Aggregate_set[Index].copy())) # index, Position, velocity
                        Local_sent[Index, 1] = True
                    Attributes[:, 8, Index, :] = [0, 0]  #set the local to 00
                    Index_par_local.pop(par)#remove from the particle index list
                    Index_par_local_set.discard(Index)
                if Local_up == L_total[1] and 2 in wall:
                    if 2 not in adhere:
                        Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up") #this function changes the velocities, center of gravity and position to propagate the 
                    else:
                        #if the wall is sticky then the whole aggregate has to loose velocity
                        Attributes[:, 8, :, :] = adhered(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_up, "up")
                        #for all of the aggregate velocity = 0, 0, if collision the velocity remains 0?
                    
                if Local_left == 0 and 1 in wall:
                    Attributes[:, 8, :, :] = bounced(Index, Attributes[:, 8, :, :], Aggregate_set[Index], Local_left, "left") #this function changes the velocities, center of gravity and position to propagate the 
                    
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
    for Index, attributes, aggregate in incoming_from_left: #for all the different particles
        #add to ghost left
        if coordinates_x == 0:
            attributes[0, 0] = attributes[0, 0] - L_total[0] #if it comes from the last proc, the position is equal to 
        if Index not in Index_par_ghost_left_set:
            Index_par_ghost_left.append(Index) #add the particle index
            Index_par_ghost_left_set.add(Index)
            Attributes[:, 1, Index, :] = attributes #associate the attributes 
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 1, Index], attributes[1, :])):
            Attributes[:, 1, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate

    #incoming from right = ghost right
    if incoming_from_right is None: #if null
        incoming_from_right = [] #set to empty to be able to loop on it without errors
    # elif incoming_from_right != [] :
    #      print("for rank : ", rank,"incoming from right : ", incoming_from_right)           
    for Index,attributes, aggregate in incoming_from_right: #for all the different particles
        #add to ghost b
        if coordinates_x == (dimensions_proc[0] - 1) :
            attributes[0, 0] = attributes[0, 0] + L_total[0] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_right_set:
            Index_par_ghost_right.append(Index)
            Index_par_ghost_right_set.add(Index)
            Attributes[:, 0, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 0, Index], attributes[1, :])):
            Attributes[:, 0, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
    
    #incoming from up = ghost up
    if incoming_from_up is None:
        incoming_from_up = []
    # elif incoming_from_up != []:
    #     print("incoming from up: ", incoming_from_up)
    for Index, attributes, aggregate in incoming_from_up:
        if coordinates_y == (dimensions_proc[1] - 1):
            attributes[0, 1] = attributes[0, 1] + L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_up_set:
            Index_par_ghost_up.append(Index)
            Index_par_ghost_up_set.add(Index)
            Attributes[:, 2, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 2, Index], attributes[1, :])):
            Attributes[:, 2, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
    
    #incoming from down = ghost down
    if incoming_from_down is None:
        incoming_from_down = []
    # elif incoming_from_down != []:
    #      print("for rank : ", rank,"incoming from down : ", incoming_from_down)
    for Index, attributes, aggregate in incoming_from_down:
        if coordinates_y == 0:
            attributes[0, 1] = attributes[0, 1] - L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_down_set:
            Index_par_ghost_down.append(Index)
            Index_par_ghost_down_set.add(Index)
            Attributes[:, 3, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 3, Index], attributes[1, :])):
            Attributes[:, 3, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
    
    #incoming from up right = ghost up right
    if incoming_from_up_right is None:
        incoming_from_up_right = []
    # elif incoming_from_up_right != []:
    #     print("incoming from up right: ", incoming_from_up_right)
    for Index, attributes, aggregate in incoming_from_up_right:
        if coordinates_x == (dimensions_proc[0] - 1):
            attributes[0, 0] = attributes[0, 0] + L_total[0] #if it comes from the first proc, te position is equal to 
        if coordinates_y == (dimensions_proc[1] - 1):
            attributes[0, 1] = attributes[0, 1] + L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_up_right_set:
            Index_par_ghost_up_right.append(Index)
            Index_par_ghost_up_right_set.add(Index)
            Attributes[:, 4, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 4, Index], attributes[1, :])):
            Attributes[:, 4, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
    
    #incoming from down right = ghost down right
    if incoming_from_down_right is None:
        incoming_from_down_right = []
    # elif incoming_from_down_right != []:
    #      print("incoming from down right: ", incoming_from_down_right)
    for Index, attributes, aggregate in incoming_from_down_right:
        if coordinates_x == (dimensions_proc[0] - 1) :
            attributes[0, 0] = attributes[0, 0] + L_total[0] #if it comes from the first proc, te position is equal to 
        if coordinates_y == 0:
            attributes[0, 1] = attributes[0, 1] - L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_down_right_set:
            Index_par_ghost_down_right.append(Index)
            Index_par_ghost_down_right_set.add(Index)
            Attributes[:, 5, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 5, Index], attributes[1, :])):
            Attributes[:, 5, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
    
    #incoming from down left = ghost down left
    if incoming_from_down_left is None:
        incoming_from_down_left = []
    # elif incoming_from_down_left != []:
    #     print("incoming from down left: ", incoming_from_down_left)
    for Index, attributes, aggregate in incoming_from_down_left:
        if coordinates_x == 0 :
            attributes[0, 0] = attributes[0, 0] - L_total[0] #if it comes from the first proc, te position is equal to 
        if coordinates_y == 0:
            attributes[0, 1] = attributes[0, 1] - L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_down_left_set:
            Index_par_ghost_down_left.append(Index)
            Index_par_ghost_down_left_set.add(Index)
            Attributes[:, 6, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate            
        elif not(np.allclose(Attributes[1, 6, Index], attributes[1, :])):
            Attributes[:, 6, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
            
    #incoming from up left = ghost up left
    if incoming_from_up_left is None:
        incoming_from_up_left = []
    # elif incoming_from_up_left != []:
    #      print("incoming from up left: ", incoming_from_up_left)
    for Index, attributes, aggregate in incoming_from_up_left:
        if coordinates_x == 0:
            attributes[0, 0] = attributes[0, 0] - L_total[0] #if it comes from the first proc, te position is equal to 
        if coordinates_y == (dimensions_proc[1] - 1):
            attributes[0, 1] = attributes[0, 1] + L_total[1] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_up_left_set:
            Index_par_ghost_up_left.append(Index)
            Index_par_ghost_up_left_set.add(Index)
            Attributes[:, 7, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
        elif not(np.allclose(Attributes[1, 7, Index], attributes[1, :])):
            Attributes[:, 7, Index, :] = attributes #associate the attributes
            Aggregate_set[Index] = aggregate
            
    XY_local_saved[t - 1, :, :] = Attributes[0, 8, :, :].copy() #save the updated values 
    
list_ranks = list(range(size))
XY_master_saved = XY_local_saved.copy()
while len(list_ranks) > 1:
    list_ranks_master = list_ranks[0::2] #even index
    list_ranks_source = list_ranks[1::2] #uneven index
    #saving the values of the local particles 
    if rank in list_ranks_master: #the proc 0 will receive all of the other locally saved positoins of particles
        #XY_master_saved = XY_local_saved.copy() #first let's say that the master positions is proc 0s and add the different position if dif than 00
        #I want to add the values dif than 0 from XY_source saved to the XY_master_saved to save the particle movment 
        source = list_ranks[list_ranks.index(rank) + 1]
        XY_source_saved = cart.recv(source = source ) #receive the send from the other proc
        for t in range (Nt): #going through all of the time dimension
            non_zero = ~np.all(XY_source_saved[t,:,:] == 0.0, axis=1)
            index_non_zero = np.flatnonzero(non_zero)#keep the index of the non zero to go through them only
            for i in range(len(index_non_zero)): #going through all of the particules
                if np.allclose(XY_master_saved[t, index_non_zero[i], :], 0.0) : #if the master value is set to 00 for now 
                    XY_master_saved[t, index_non_zero[i], :] = XY_source_saved[t, index_non_zero[i], :].copy() #replace the value in the master for the source value,
                elif not np.allclose(XY_master_saved[t, index_non_zero[i], :], XY_source_saved[t, index_non_zero[i], :]): # check if 2 dif value dif than 00 exist for 2 dif proc
                    dx = XY_master_saved[t, index_non_zero[i], 0] - XY_source_saved[t, index_non_zero[i], 0]
                    dy = XY_master_saved[t, index_non_zero[i], 1] - XY_source_saved[t, index_non_zero[i], 1]
                    remainer_x = dx % L_total[0] 
                    remainer_y = dy % L_total[1]
                    if not (np.allclose(remainer_x, 0.0) or np.allclose(remainer_x, L_total[0])) and (np.allclose(remainer_y,0.0) or np.allclose(remainer_y,L_total[1])) :
                        print("The position ",XY_source_saved[t, index_non_zero[i], :] , " for particule ", index_non_zero[i], " in proc: ", source, " is different than in the master at t = ", t, "\n", " which is : ", XY_master_saved[t, index_non_zero[i], :])        
    elif rank in list_ranks_source:
        dest = list_ranks[list_ranks.index(rank) - 1]
        cart.send(XY_master_saved, dest = dest)
    list_ranks = list_ranks_master
    comm.barrier()
    
print(rank)
if rank == 0:
    non_zero = ~np.all(XY_master_saved[Nt-1,:,:] == 0., axis=1)
    XY_count = XY_master_saved[Nt-1, non_zero]
    Num_Particules_end_count = len(XY_count)
    dif_num = abs(Num_Particules_end_count - Num_Particules_end)                               
    if  dif_num != 0:
        print("OH NO, there are ", dif_num," missing particles ")

#WAIT ON THE OTHER PROC TO FINISH
comm.Barrier()
t1 = time.perf_counter()

#function to have the display particles be of comparable size
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
    
    #plot the end surface aggregate
    surface_aggregate, axis = plt.subplots()
    axis.set_xlim(0, 20)
    axis.set_ylim(L_total[1] - 1, L_total[1]) #just want to see the top part
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    si = radius_to_s(axis, Radius_molecule) # understandable size of particles - good scale
    plot = axis.scatter(XY_master_saved[Nt - 1, :, 0].copy(), XY_master_saved[Nt - 1, :, 1].copy(), s = si ) #plot itself
    title_plot = axis.set_title(f"Surface aggregation at t = {t}") #title
    axis.set_aspect("equal", adjustable="box")  #realistic aspect ratio
    surface_aggregate.tight_layout()
    surface_aggregate.savefig("surface_aggregation.png", dpi = 500, bbox_inches = "tight")
    plt.close(surface_aggregate) #close so that it doesnt count in the runtime.
    
    
    # plot
    # --- Figure and initial scatter ---
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
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
        writer = FFMpegWriter(fps = fps, codec = "libx264", bitrate = -1, extra_args=[
            "-preset", "slow",        # better compression at same quality
            "-crf", "17",             # high quality (try 16–18)
            "-pix_fmt", "yuv420p"     # compatibility
        ])
        ani.save("particle_animation.mp4", writer=writer)
        

    plt.close(fig)#not showing but saving
    #time print
    t2 = time.perf_counter()
    print("For ", size, ", the runtime before the mp4 is: ", t1 - t0)
    print("For ", size, ", the runtime with the mp4 is : ", t2 - t0)
       
MPI.Finalize() # stop the parallelization




