#I am wanting to code a mpi code that can send one particule from one proc to another - 1D
from mpi4py import MPI #mpi
import numpy as np #np calculation
import matplotlib.pyplot as plt #plotting
from matplotlib.animation import FuncAnimation, FFMpegWriter #animation
plt.clf()

#mpi init the domain of mpi - DO NOT TOUCH
comm = MPI.COMM_WORLD #INIT the mpi
rank = comm.Get_rank() #get the processor rank
size = comm.Get_size() #get the total num of processors
if rank > 0:
    left = rank - 1 
else: 
    left = size - 1
if rank < size - 1:
    right = rank + 1
else: 
    right = 0

# save animation as a GIF? - To SET 
save_gif_animation = True

#time - TO SET
T =  8# seconds to change
dt = 0.05 #delta t 
Nt = int(T/ dt) #num of Iterations 

#Mesh - TO SET
L_total = np.array([20, 20]) #Total Size in microm

#Particules - TO SET
Num_Particules = 500 #for now the second particule doesnt move but is needed to code
XY_start = np.empty((Num_Particules,2)) #particule start position 

XY_start[:, 0] = np.random.uniform(low = 0, high = L_total[0], size = Num_Particules) #rand the position
XY_start[:, 1] = np.random.uniform(low = 0, high = L_total[1], size = Num_Particules)


Vp = np.zeros((Num_Particules,2))
Vp[:, 0] = np.random.uniform(low = -1.3, high = 1.3, size = Num_Particules)


#Local MESH - DO NOT TOUCH
Local_width = np.array([L_total[0]/ size , L_total[1]]) # Width of 1 area and height
Buffer_zone_width = np.array([abs(Vp[0,0]) * dt * 2 , L_total[1]]) #Buffer depend on the velocity of the particule, for now in 1 D - to change for 2D

#Each Proc - DO NOT TOUCH
Local_start = np.array([0]) # defining the start before being rewritten
Local_ghost_start = np.array([0])
Local_end = np.array([0]) # defining the end before being rewritten
Local_ghost_end = np.array([0])

Local_start = (rank * Local_width[0]) # the principal zone starts
Local_ghost_start = Local_start - Buffer_zone_width[0] #the ghost zone starts before
Local_end = (rank + 1 ) * Local_width[0] #the prinicpal zone ends
Local_ghost_end = Local_end + Buffer_zone_width[0] #the ghost zone ends
    
    
#checks - print if you want - prints only once for proc
# print("Local Start for processor ", rank, " is : ", Local_start, "\n")
# print("Local Ghost Start for processor ", rank, " is : ", Local_ghost_start, "\n")
# print("Local End for processor ", rank, " is : ", Local_end, "\n")
# print("Local Ghost End for processor ", rank, " is : ", Local_ghost_end, "\n")

# Particule position - to change - set to empty for now- pay attention in case some particules have a 00 position and 00 velocity might not work
XY_local = np.zeros((Num_Particules,2))
XY_ghost_a = np.zeros((Num_Particules,2))
XY_ghost_b = np.zeros((Num_Particules,2))
# Particule speeds- set to empty for now
Vp_local = np.zeros((Num_Particules,2))
Vp_ghost_a = np.zeros((Num_Particules,2))
Vp_ghost_b = np.zeros((Num_Particules,2))

#setting the particule index key
#num of particules per processor calculated thanks to len (Index)
Index_par_local = []
Index_par_local_set = set()
Index_par_ghost_a = []
Index_par_ghost_a_set = set()
Index_par_ghost_b = []
Index_par_ghost_b_set = set()
 
#Initialisation 
# of particules placement, keeping track of which particule is where (kind of like a master key)
for p in range(Num_Particules): 
    if Local_start <= XY_start[p, 0] < Local_end : #for now the processor has a particule if it is in
        Index_par_local.append(p) #know whch particule I have, helps not printing the same particle twice
        Index_par_local_set.add(p)
        XY_local[p, :]= XY_start[p, :]
        Vp_local[p, :]= Vp[p, :] * dt #movment during one dt ( = distance in one dt)
    elif Local_ghost_start <= XY_start[p, 0] < Local_start:
        Index_par_ghost_a.append(p)
        Index_par_ghost_a_set.add(p)
        XY_ghost_a[p, :] = XY_start[p, :]
        Vp_ghost_a[p, :] = Vp[p, :] * dt #movment during one dt ( = distance in one dt)
    elif Local_end <= XY_start[p, 0] <= Local_ghost_end:
        Index_par_ghost_b.append(p)
        Index_par_ghost_b_set.add(p)
        XY_ghost_b[p, :] = XY_start[p, :]
        Vp_ghost_b[p, :] = Vp[p, :] * dt #movment during one dt ( = distance in one dt)

# for 2d start to change here 

#Initialisation of the Update of the local and ghost locations
XY_local_update = XY_local.copy()
XY_ghost_a_update = XY_ghost_a.copy()
XY_ghost_b_update = XY_ghost_b.copy()


#initiate the saving of information
#I wonder if I should print the particles in the ghost areas as well in different colors... For now, only the local xy will be printed
XY_master_saved = np.zeros((Nt, Num_Particules, 2)) #use at the end when merging the info
XY_local_saved = np.zeros((Nt, Num_Particules, 2)) #saving the positions of every particle of each processor in the time loop, will merge the info at the end
XY_local_saved[0,:,:] = XY_local.copy()


#Boolean value that will keep track of if i have already sent a particule to another proc so 
# that it doesnt resend it, will be reset to false when the particule leaves the proc t enable rollback
Local_sent_next = np.zeros((Num_Particules), dtype = bool)
Local_sent_prev = np.zeros((Num_Particules), dtype = bool)

for t in range(1, Nt+ 1):   
    #case where the particule is inside the local area ot the ghost    
    Particle_info_right = [] #list of particles to send to r + 1 
    Particle_info_left = [] #list of particles to send to r - 1
    if len(Index_par_local) > 0 or len(Index_par_ghost_a) > 0 or len(Index_par_ghost_b) > 0 :
        #there is one particule inside the whole processor realm so we update their position; if no particule in the area it should return 00
        XY_local = XY_local_update #initializing the old value 
        XY_local_update = XY_local + Vp_local #new value = old + distance in one frame
        XY_ghost_a = XY_ghost_a_update
        XY_ghost_a_update = XY_ghost_a + Vp_ghost_a
        XY_ghost_b = XY_ghost_b_update
        XY_ghost_b_update = XY_ghost_b + Vp_ghost_b
        #update the local first if it is out of bounds 
        for par in reversed(range(len(Index_par_local))):
            Index = Index_par_local[par] #index in the xy, vp and other
            Position = XY_local_update[Index, 0]
            #moving forward technically but also called when moving backward
            if (Position >= (Local_end - Buffer_zone_width[0]) and not Local_sent_next[Index]): #xy >= ghost a du prochain
                #The particule entered the ghost zone of the next processor, 
                #we send it to the next proc if we havent yet
                Particle_info_right.append((Index, XY_local_update[Index,:].copy(), Vp_local[Index,:].copy())) # index, position, velocity
                Local_sent_next[Index] = True
            elif ((Position < (Local_end - Buffer_zone_width[0]) or Position > Local_ghost_end )and Local_sent_next[Index]): #if it leaves on the other side back to false
                #print("The particule has left the proc")
                Local_sent_next[Index] = False
            elif Position >= Local_end:
                #the particule left the main local proc area, enters the ghost are    
                Index_par_ghost_b.append(Index)#add the index to the end of the list  
                Index_par_ghost_b_set.add(Index)        
                XY_ghost_b_update[Index,:] = XY_local_update[Index, :].copy() #associate the local position with the ghost         
                Vp_ghost_b[Index, :] = Vp_local[Index, :].copy() #associate the local speed with the ghost 
                XY_local_update[Index, :] = [0, 0] #set the local to 00 
                Vp_local[Index, :] = [0, 0] # set the local back to 00
                Index_par_local.pop(par)#remove from the particle index list
                Index_par_local_set.discard(par)
            
            #rollback
            elif (Position <= (Local_start + Buffer_zone_width[0]) and not Local_sent_prev[Index]):
                #The particule entered the ghost zone of the "previous" processor,
                #we send it to the prev proc if we havent yet
                #print("position : ", XY_local_update[Index, 0], " proc: ", rank," start", Local_start, " buffer", Buffer_zone_width ) #check the particules sent
                Particle_info_left.append((Index, XY_local_update[Index,:].copy(), Vp_local[Index,:].copy())) # index, position, velocity
                Local_sent_prev[Index] = True
            elif ((Position > (Local_start + Buffer_zone_width[0]) or Position < Local_ghost_start )and Local_sent_prev[Index]): #reset the boolean to ensure the particule can be sent again
                #print("The particule has left the proc") 
                Local_sent_prev[Index] = False
            elif XY_local_update[Index, 0] < Local_start:
                #the particule left the main local proc area, enters the ghost a area    
                Index_par_ghost_a.append(Index)#add the index to the end of the list
                Index_par_ghost_a_set.add(Index)         
                XY_ghost_a_update[Index,:] = XY_local_update[Index, :].copy() #associate the local position with the ghost         
                Vp_ghost_a[Index, :] = Vp_local[Index, :].copy() #associate the local speed with the ghost 
                XY_local_update[Index, :] = [0, 0] #set the local to 00 
                Vp_local[Index, :] = [0, 0] # set the local back to 00
                Index_par_local.pop(par)#remove from the particle index list
                Index_par_local_set.discard(par)  
        for par_b in reversed(range(len(Index_par_ghost_b))):
            Index = Index_par_ghost_b[par_b] #index of the particule in the ghost b area to use for the xy vp which are indexed following the grand scheme to obliviate confusion
            #2 cases for now forward or backward
            #forward b to leave
            if XY_ghost_b_update[Index, 0] > Local_ghost_end : 
                #particule is leaving the ghost b area 
                XY_ghost_b_update[Index, :] = [0, 0] #technically, I already sent it so no send
                Vp_ghost_b[Index, :] = [0, 0]
                Index_par_ghost_b.pop(par_b) #remove the particule from the count
                Index_par_ghost_b_set.discard(par_b)
            #roll back b to local
            elif XY_ghost_b_update[Index, 0] < Local_end:
                #the particule enters the local area and leaves the ghost b area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_local_update[Index, :] = XY_ghost_b[Index, :].copy() #update the local position of the particule
                Vp_local[Index, :] = Vp_ghost_b[Index, :].copy() #Update the local speed of the particule
                XY_ghost_b_update[Index, :] = [0, 0] #Remove the positon of the particule
                Vp_ghost_b[Index, :] = [0, 0] #Remove the speed of the particule 
                Index_par_ghost_b.pop(par_b) #Index of the new particule added
                Index_par_local_set.discard(par_b)
            
        for par_a in reversed(range(len(Index_par_ghost_a))):
            Index = Index_par_ghost_a[par_a] #index of the particule in the ghost a area
            #2 cases froward or backward
            #forward a to local 
            if XY_ghost_a_update[Index, 0] >= Local_start:
                #the particule enters the local area and leaves the ghost a area
                Index_par_local.append(Index) #add the index to the list
                Index_par_local_set.add(Index)
                XY_local_update[Index, :] = XY_ghost_a[Index, :].copy() #update the local position of the particule
                Vp_local[Index, :] = Vp_ghost_a[Index, :].copy() #Update the local speed of the particule
                XY_ghost_a_update[Index, :] = [0, 0] #Remove the positon of the particule
                Vp_ghost_a[Index, :] = [0, 0] #Remove the speed of the particule 
                Index_par_ghost_a.pop(par_a) #Index of the new particule added
                Index_par_ghost_a_set.discard(par_a)
            #rollback a to leave
            elif XY_ghost_a_update[Index, 0] < Local_ghost_start : 
                #particule is leaving the ghost a area 
                XY_ghost_a_update[Index, :] = [0, 0] #technically, I already sent it so no send
                Vp_ghost_a[Index, :] = [0, 0]
                # if Local_sent_prev[Index] == True:
                #     Local_sent_prev[Index] = False #set back to False
                # else :
                #     print("The particule was never sent to another proc, it is lost. \n")
                Index_par_ghost_a.pop(par_a) #remove the particule from the count
                Index_par_ghost_a_set.discard(par_a) 
 
    #do the comms now, so that it send the whole list
    incoming_from_left = comm.sendrecv( sendobj = Particle_info_right, dest = right, sendtag = 0, source = left, recvtag = 0)
    #sending a particule to the right, the tag helps identifying hte different sends (if it is going left or right)
    incoming_from_right = comm.sendrecv( sendobj = Particle_info_left, dest = left, sendtag = 1, source = right, recvtag = 1)
    #sending a particule to the left, tag = 1 for right to left
    
    #how to deal with the new data
    if incoming_from_left is None: #if null
        incoming_from_left = [] #set to empty to be able to loop on it without errors
    # else :
    #     print("incoming from left: ", incoming_from_left)
    
    for Index, pos, vel in incoming_from_left: #for all the different particles
        #add to ghost a
        if rank == 0:
            pos[0] = pos[0] - L_total[0] #if it comes from the last proc, the position is equal to 
        if Index not in Index_par_ghost_a_set:
            Index_par_ghost_a.append(Index) #add the particle index
            Index_par_ghost_a_set.add(Index)
            XY_ghost_a_update[Index] = pos #associate the postion
            Vp_ghost_a[Index] = vel #and velocity

    
    if incoming_from_right is None: #if null
        incoming_from_right = [] #set to empty to be able to loop on it without errors
    # else :
    #     print("incoming from right : ", incoming_from_right)
                
    for Index, pos, vel in incoming_from_right: #for all the different particles
        #add to ghost b
        if rank == size - 1:
            pos[0] = pos[0] + L_total[0] #if it comes from the first proc, te position is equal to 
        if Index not in Index_par_ghost_b_set:
            Index_par_ghost_b.append(Index)
            Index_par_ghost_b_set.add(Index)
            XY_ghost_b_update[Index] = pos
            Vp_ghost_b[Index] = vel
            
    XY_local_saved[t - 1, :, :] = XY_local

if rank == 0: #the proc 0 will receive all of the other locally saved positoins of particles
    XY_master_saved = XY_local_saved.copy() #first let's say that the master positions is proc 0s and add the different position if dif than 00
    #I want to add the values dif than 0 from XY_source saved to the XY_master_saved to save the particle movment 
    for source in range(1,size): #go thru all of the proc
        XY_source_saved = comm.recv(source = source) #receive the send from the other proc
        for t in range (Nt): #going through all of the time dimension
            for i in range(Num_Particules): #going through all of the particules
                if not np.allclose(XY_source_saved[t, i, :], 0.0):#if dif than null which is 00 for now, using this instead of direct comparaison beause there might be floats
                    if np.allclose(XY_master_saved[t, i, :], 0.0) : #if the master value is set to 00 for now 
                        XY_master_saved[t, i, :] = XY_source_saved[t, i, :].copy() #replace the value in the master for the source value,
                    elif not np.allclose(XY_master_saved[t, i, :], XY_source_saved[t, i, :]): # check if 2 dif value dif than 00 exist for 2 dif proc
                        print("The position ",XY_source_saved[t, i, :] , " for particule ", i, " in proc: ", source, " is different than in the master at t = ", t, "\n", " which is : ", XY_master_saved[:,i,t])
                                   
           
else :
    comm.send(XY_local_saved, dest = 0)


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
    
    #num particules - start with empty data
    scat = ax.scatter( np.zeros( Num_Particules), np.zeros( Num_Particules), s=50)
    #define the title before
    title = ax.set_title("Particle animation")
    
    # --- choose video fps + how many sim steps per video frame ---
    fps = 30
    stride = max(1, int((1 / fps) / dt))   # sim steps per rendered frame
    frames = range(0, Nt, stride)

    #realistic aspect ratio
    ax.set_aspect("equal", adjustable="box")
    
    for k in range(size-1):
        ax.axvline(x = (k + 1) * (L_total[0]/size), color = "red", linestyle = "--", linewidth = 1)

    def update(frame):  #change to have n particules 
        positions = XY_master_saved[frame, :, :] 
        # set_offsets expects an array of shape (n_points, 2)
        scat.set_offsets(positions)
        current_time = frame * dt
        ax.set_title(f"Particle animation (t = {current_time:.3f} s)")

    # --- Create animation ---  #change to have n particules 
    ani = FuncAnimation( fig, update, frames = frames, interval=1000/fps, blit = False)
    if save_gif_animation:
        writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800
    )
    ani.save("particle_animation.mp4", writer=writer)

    plt.close(fig)#not showing but saving
       
MPI.Finalize # stop the parallelization

#check that all particles are still there



