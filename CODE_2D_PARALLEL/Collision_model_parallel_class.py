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
from particle_interactions import update_particles_aggregation, update_particles_collision, broad_detect, narrow_detect
from Particle import Particle

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
T_add_particles = 0.5 #time for which I add particles for - HERE
dt = 0.05 #delta t 
#mp4 animation
save_gif_animation = True # save animation as a mp4? - To SET 
#Mesh - TO SET
L_total = np.array([20, 20]) #Total Size in microm - HERE
#Particles - TO SET
position = 0 #0 for auto and 1 for manual choice
Num_Particules = 400 #particles to start - HERE
Num_Particules_dt= 0 #particls added per second - HERE
#TiO2 properties - rutile for now
A_h = 6*10**(-20) #hamaker constant for rutile Tio2
Radius_molecule = 0.1 #radius of the particule in micrometer 
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
# wall.add(0)
# wall.add(1)
#bounciness: is the wall bouncy?  #0 for right, 1 for left, 2 for up, 3 for bottom, 4 for none
bounce = set() 
bounce.add(2)
# bounce.add(3)
# bounce.add(0)
# bounce.add(1)

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
if rank == 1: #do not overload proc 0, need the if so that the rand doesnt run for each pro
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
Particles_local = [] #initialise the particles
Particles_left = []
Particles_right = []
Particle_up = []
Particle
for p in range(Num_Particules_end): 
    Position_x = XY_start[p, 0]
    Position_y = XY_start[p, 1]
    if Local_left <= Position_x < Local_right and Local_down <= Position_y < Local_up: #for now the processor has a particule if it is in
        Index_par_local.append(p) #know whch particle I have, helps not printing the same particle twice
        Index_par_local_set.add(p)
        Particles_local = [Particles_local , Particle(index = p, mass = Mass_one_particle, radius = Radius_molecule, position = XY_start[p, :], velocity = Vp[p, :], show = Added_par[p])]
    elif Local_ghost_left <= Position_x < Local_left and Local_down <= Position_y < Local_up: #particule in the left ghost
        Index_par_ghost_left.append(p)
        Index_par_ghost_left_set.add(p)
        Particles_left
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
         