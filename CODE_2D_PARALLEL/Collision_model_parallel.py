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
Num_Particules = 2 #for now the second particule doesnt move but is needed to code
XY_start = np.zeros((2,2)) #particule start position, rand later
XY_start[0,:] = [0.2, 10]
XY_start[1,:] = [4, 10]
Vp = np.zeros((2,2))
Vp[0, :] = [0.5, 0] # Velocity of the particule 1
Vp[1, :] = [0, 0]
print(XY_start[1,0])

#Local MESH - DO NOT TOUCH
Local_width = np.array([L_total[0]/ size , L_total[1]]) # Width of 1 area and height
Buffer_zone_width = np.array([Vp[0,0] * dt * 2 , L_total[1]]) #Buffer depend on the velocity of the particule, for now in 1 D - to change for 2D

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

#num of particules per processor
Local_num_particules = 0 #define the number of particules before they are sorted in the different processors
Ghost_num_particules_a = 0
Ghost_num_particules_b = 0

#Particule movment - to move once we have different particules in different zones (move in the for loop)

# Particule position - to change - set to empty for now- pay attention in case some particules have a 00 position and 00 velocity might not work
# XY_local = np.empty( 2, Num_Particules)
XY_local = np.empty((2,Num_Particules))
XY_ghost_a = np.empty((2,Num_Particules))
XY_ghost_b = np.empty((2,Num_Particules))
# Particule speeds- set to empty for now
Vp_local = np.empty((2,Num_Particules))
Vp_ghost_a = np.empty((2,Num_Particules))
Vp_ghost_b = np.empty((2,Num_Particules))

 
#Initialisation of particules placement
for p in range(Num_Particules): 
    print(p)
    if Local_start <= XY_start[p, 0] < Local_end : #for now the processor has a particule if it is in
        Local_num_particules = Local_num_particules + 1 # append by the particule in the zone
        XY_local[p, :]= XY_start[p, :]
        Vp_local[p, :]= Vp[p, :] * dt #movment during one dt ( = distance in one dt)
    elif Local_ghost_start <= XY_start[p, 0] < Local_start:
        Ghost_num_particules_a = Ghost_num_particules_a + 1
        XY_ghost_a[p, :] = XY_start[p, :]
        Vp_ghost_a[p, :] = Vp[p, :] * dt #movment during one dt ( = distance in one dt)
    elif Local_end < XY_start[p, 0] <= Local_ghost_end:
        Ghost_num_particules_b = Ghost_num_particules_b + 1
        XY_ghost_b[p, :] = XY_start[p, :]
        Vp_ghost_b[p, :] = Vp[p, :] * dt #movment during one dt ( = distance in one dt)

#works for 2 particules for now up to here

#Updating the local and ghost locations
XY_local_update = XY_local
XY_ghost_a_update = XY_ghost_a
XY_ghost_b_update = XY_ghost_a

#Boolean value that will keep track of if i have already sent a particule to another proc so 
# that it doesnt resend it, will be reset to false when the particule leaves the proc t enable rollback
Local_sent_next = np.zeros((Num_Particules,1), dtype=bool)

for t in range(Nt):
    
    data = comm.recv(source = rank - 1) #deal with other cases later aligator
    #If i receive data :
        #if data [3] = "ghost a"
            #Ghost_num_particules_a += 1
            #XY_ghost_a[]
        #else if data[3] = "ghost b"
            #Ghost_num_particules_a += 1
            #XY_ghost_b =
        #else
            #break :error
    #case where the particule is inside the local area ot the ghost    
    if Local_num_particules > 0 or Ghost_num_particules_a > 0 or Ghost_num_particules_b:
        #there is one particule inside the whole processor realm so we update their position; if no particule in the area it should return 00
        XY_local = XY_local_update
        XY_local_update = XY_local + Vp_local
        XY_ghost_a = XY_ghost_a_update
        XY_ghost_a_update = XY_ghost_a + Vp_ghost_a
        XY_ghost_b = XY_ghost_b_update
        XY_ghost_b_update = XY_ghost_b + Vp_ghost_b
        #update the local first if it is out of bounds
        for par in range(Local_num_particules):
            if XY_local_update[par, 0] >= (Local_start - Buffer_zone_width) and Local_sent_next == False:
                #The particule entered the ghost zone of the next processor, roll back to plan
                #we send it to the next proc if we havent yet
                Particle_info = np.array([par, XY_local_update[par,:], Vp_local[par,:], "ghost_a"]) #index, position, velocity, ghost area 
                comm.send(Particle_info, dest = (rank + 1))#for now it is only going to rank + 1
                Local_sent_next[par, 0] = True
            elif XY_local_update[par, 0] > Local_end:
                Local_num_particules = Local_num_particules - 1
                XY_local_update[par, :] = [0, 0]
            
    
    
    


    
MPI.Finalize