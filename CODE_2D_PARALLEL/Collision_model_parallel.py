#I am wanting to code a mpi code that can send one particule from one proc to another
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD #INIT the mpi
rank = comm.Get_rank() #get the processor rank
size = comm.Get_size() #get the total num of processors

#time - TO SET
T = 10 # seconds to change
dt = 0.0005 #delta t 
Nt = int(T/ dt) #num of Iterations 

#Mesh - TO SET
L_total = np.array([8, 20]) #Total Size in microm

#Particules - TO SET
Num_Particules = 1 #for now 1
XY_start = np.array([ 0.2, 10]) #particule start position, rand later
Vp = np.array([ 0.5, 0]) # Velocity of the particule


#Local MESH - DO NOT TOUCH
Local_width = np.array([L_total[0]/ size , L_total[1]]) # Width of 1 area and height
Buffer_zone_width = np.array([Vp[0] * dt * 2 , L_total[1]]) #Buffer depend on the velocity of the particule, for now in 1 D - to change for 2D

#Each Proc - DO NOT TOUCH
Local_start = np.array([0]) # defining the start before being rewritten
Local_ghost_start = np.array([0])
Local_end = np.array([0]) # defining the end before being rewritten
Local_ghost_end = np.array([0])

if rank == 0 :
    Local_start = 0 # mesh starts at 0
    Local_ghost_start = 0 #mesh starts at 0 so no buffer
else:
    Local_start = (rank * Local_width[0]) # the principal zone starts
    Local_ghost_start = Local_start - Buffer_zone_width[0] #the ghost zone starts before
    
if rank == (size - 1):
    Local_end = L_total[0] #mesh ends at the L_total [0]
    Local_ghost_end = Local_end #end so no buffer
else:
    Local_end = (rank + 1 ) * Local_width[0] #the prinicpal zone ends
    Local_ghost_end = Local_end + Buffer_zone_width[0] #the ghost zone ends
    
    
#checks
print("Local Start for processor ", rank, " is : ", Local_start, "\n")
print("Local Ghost Start for processor ", rank, " is : ", Local_ghost_start, "\n")
print("Local End for processor ", rank, " is : ", Local_end, "\n")
print("Local Ghost End for processor ", rank, " is : ", Local_ghost_end, "\n")

Local_num_particules = 0 #define the number of particules before they are sorted in the different processors
Ghost_num_particules_a = 0
Ghost_num_particules_b = 0

#Particule movment - to move once we have different particules in different zones (move in the for loop)

# Particule position - to change - set to 0 for now
# XY_local = np.zeros( 2, Num_Particules)
# Particule speeds- set to zeros fro now
# Vp_dt = np.zeros( 2, Num_Particules)
XY_local = np.empty(2)
XY_ghost_a = np.empty(2)
XY_ghost_b = np.empty(2)
Vp_local = np.empty(1)
Vp_ghost_a = np.empty(1)
Vp_ghost_b = np.empty(1)
 
#Initialisation of particules placement
for p in range(Num_Particules):
    if Local_start < XY_start[0] < Local_end :
        Local_num_particules = Local_num_particules + 1 # append by the particule in the zone
        XY_local[2] = XY_start
        Vp_local = Vp * dt #movment during one dt ( = distance in one dt)
    if Local_ghost_start < XY_start[0] < Local_start:
        Ghost_num_particules_pre = 0
        XY_ghost_a[2] = XY_start
        Vp_ghost_a = Vp * dt #movment during one dt ( = distance in one dt)
    if Local_end < XY_start[0] < Local_ghost_end:
        Ghost_num_particules_post = 0
        XY_ghost_b[2] = XY_start
        Vp_ghost_b = Vp * dt #movment during one dt ( = distance in one dt)
 


for t in range(Nt):
    
    data = comm.recv(source = rank - 1) #deal with other cases later aligator
    #If i receive data :
        #if data [3] = "ghost a"
            #Ghost_num_particules_a += 1
        #else if data[3] = "ghost b"
            #Ghost_num_particules_a += 1
        #else
            #break :error
    

        
    if Local_num_particules > 0:
        XY_local[1] = XY_local[2]
        XY_local[2] = XY_local[1] + Vp_local
        
        if XY_local [2] > (Local_start - Buffer_zone_width):
            #The particule entered the ghost zone of the next processor, roll back to plan
            Particle_info = np.array([XY_local [2], Vp_local, "ghost_a"])
            comm.send(Particle_info, dest = (rank + 1))
            
    
    
    


    
MPI.Finalize