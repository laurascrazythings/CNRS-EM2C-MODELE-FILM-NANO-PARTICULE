import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy import interpolate
import datetime as dtm
import csv

import Calculation

### Initialization ###

simulation_time = 5# Total simulation time

# Runtime calculation
beginning_date_and_time = dtm.datetime.now()


# Animation saving
SAVE_ANIMATION = False # Constant to control animation saving

# Position saving
SAVE_POSITION = True # Constant to control position saving in a csv

# Print kinetic energy
PRINT_KINETIC_ENERGY = False # Constant to control kinetic energy printing


# Parameters
Np = 2000  # Number of particles
radius_diameter = 10**(-2) #Radius of a particule (µm)
collision_treshold = 2 * radius_diameter # Collision distance threshold (µm)
diffusion_coefficient = 0
stick_probability = 2/3
Lx = 4  # Domain width (µm)
Ly = 20  # Domain height (µm)


# Initialize particle positions and velocities
Xp = Lx * np.random.rand(Np)  # Random X positions
Yp = Ly * np.random.rand(Np)  # Random Y positions
Xp_total = [Xp] # Matrix of X positions for all Tout
Yp_total = [Yp] # Matrix of Y positions for all Tout

Up = 2 * np.random.rand(Np) - 1  # Random X velocities (Up in [-1, 1])
Vp = 2 + 2 * np.random.rand(Np) - 1  # Random Y velocities (Vp in [1, 3])
Up_total = [Up] # Matrix of X velocities for all Tout
Vp_total = [Vp] # Matrix of Y velocities for all Tout


# Place fixed particles at the top of the domain
Npsol = int(Lx / collision_treshold)  # Number of fixed particles
Yp[:Npsol] = Ly - collision_treshold / 2  # Fix the Y position of the fixed particles
Xp[:Npsol] = np.linspace(collision_treshold / 2, Lx - collision_treshold / 2, Npsol)  # Distribute X positions evenly


# Time settings
dt = 0.0005  # Time step
Nt = int(simulation_time / dt)  # Number of time steps

# Animation parameters
Tout = 0.01  # Time interval for animation output
Toutput = Tout # First time of output

# Add particle parameters
Tadd = 0.5
Tadding = Tadd

# Stuck array
stuck = np.zeros(Np, dtype=bool)  # Array to track fixed particles (stuck[i] = True if particle stuck, else stuck[i] = False)
stuck[:Npsol] = True  # Mark the top row of particles as fixed


# Thickness of the film initialization (Interpolation)
number_points_interpolation = 50 # Number of points used for interpolation
interval_interpolation_size = Lx / number_points_interpolation # Sized of the interval in which we select a single point for interpolation
x_abscissa = np.linspace(0, Lx, 100) # Abscissa for the interpolation function 

s_parameter_Bspline = 0.1 # Parameter of the B-spline : smoothing factor
k_parameter_Bspline = 3 # Parameter of the B-spline : degree 

X_coordinate_interpolation_animation = [np.linspace(0, Lx, number_points_interpolation+1)] # Points used for interpolation
Y_coordinate_interpolation_animation = [np.linspace(Ly - collision_treshold / 2, Ly - collision_treshold / 2, number_points_interpolation+1)] # Points used for interpolation

X_interpolated_animation = [x_abscissa] # Interpolation function
Y_interpolated_animation =[np.linspace(Ly - collision_treshold / 2, Ly - collision_treshold / 2, 100)] # Interpolation function

film_thickness = [collision_treshold / 2]

# Total kinetic energy calculation
if PRINT_KINETIC_ENERGY:

    kinetic_energy = 0
    for index in range(len(Vp)):

        kinetic_energy += np.sqrt(Up[index]**2 + Vp[index]**2)

    print(f"Énergie cinétique totale = {kinetic_energy:.2f} J")
    #print("---------------------")

### Functions ###

def detect_collisions(Xp, Yp, d):

    """ Detect particle collisions using a KD-Tree for efficiency.

    Args:
        Xp (array): Array of particle X positions.
        Yp (array): Array of particle Y positions.
        d (float): Collision threshold distance.

    Returns:
        array: Pairs of indices representing colliding particles. """

    tree = cKDTree(np.c_[Xp, Yp])  # Build a KD-Tree from particle positions
    pairs = tree.query_pairs(d)  # Find all pairs of particles within distance d
    
    return np.array(list(pairs))


@jit(nopython=True)
def handle_collisions(coll_pairs, Xp, Yp, Up, Vp, stuck):

    """ Update the velocities of particles after a collision

    Args:
        coll_pairs (array): Pairs of indices representing colliding particles.
        Xp, Yp (array): Particle positions.
        Up, Vp (array): Particle velocities.
        stuck (array): Boolean array indicating whether particles are fixed. """

    for i, j in coll_pairs: # Browsing of each collision

        kij = np.array([Xp[j] - Xp[i], Yp[j] - Yp[i]])  # Relative position vector
        kij /= np.linalg.norm(kij)  # Normalize the vector

        wij = np.array([Up[j] - Up[i], Vp[j] - Vp[i]])  # Relative velocity vector
        wijk = np.dot(kij, wij)  # Project relative velocity onto the collision axis

        if wijk < 0:  # Only process collisions if particles are approaching

            if not stuck[i] and not stuck[j]:  # Both particles are free to move
                Up[i] += wijk * kij[0]
                Vp[i] += wijk * kij[1]
                Up[j] -= wijk * kij[0]
                Vp[j] -= wijk * kij[1]

            elif stuck[i] and stuck[j]:  # Both particles are fixed

                stuck[i] = stuck[j] = True  # Mark both as stuck
                Up[i] = Vp[i] = 0  # Set velocities to zero
                Up[j] = Vp[j] = 0
            
            elif stuck[i]: # Particle i is stuck but particle j is free to move

                if np.random.uniform(0,1) < stick_probability: # Adhesion of j on i

                    stuck[i] = stuck[j] = True  # Mark both as stuck
                    Up[i] = Vp[i] = 0  # Set velocities to zero
                    Up[j] = Vp[j] = 0
                
                else: # No adhesion

                    Up[i] = Vp[i] = 0  # Set velocities to zero for the stuck particle
                    Up[j] -= wijk * kij[0] # The other particle is free to move
                    Vp[j] -= wijk * kij[1]
            
            else: # Particle j is stuck but particle i is free to move
                
                if np.random.uniform(0,1) < stick_probability: # Adhesion of i on j

                    stuck[i] = stuck[j] = True  # Mark both as stuck
                    Up[i] = Vp[i] = 0  # Set velocities to zero
                    Up[j] = Vp[j] = 0
                
                else: # No adhesion

                    Up[j] = Vp[j] = 0 # Set velocities to zero for the stuck particle
                    Up[i] += wijk * kij[0] # The other particle is free to move
                    Vp[i] += wijk * kij[1] # The other particle is free to move


@jit(nopython=True)
def find_interpolation_points (Xp, Yp, stuck, number_points_interpolation, interval_interpolation_size): 

    """ Determine all the needed points for interpolation (used to calculate the film thicnkess later)"""

    #List of coordinates of stuck particles
    X_stuck, Y_stuck = [], []
        
    for index, element in enumerate(stuck):
        if element == True:
            X_stuck.append(Xp[index]), Y_stuck.append(Yp[index])

    # List of points used for interpolation
    X_coordinate_interpolation, Y_coordinate_interpolation = np.zeros(number_points_interpolation), np.zeros(number_points_interpolation)

    # Scroll through each interval to find the interpolation point (ie the lowest one)
    for point in range(0, number_points_interpolation ):

        # Boundary of the current interval
        dmin = (point)*interval_interpolation_size
        dmax = (point + 1) * interval_interpolation_size
        
        # List of all the stuck parti in the current interval
        X_in_interval, Y_in_interval = [], []

        for index, element in enumerate(X_stuck): 
            if element > dmin and element < dmax:
                X_in_interval.append(element)
                Y_in_interval.append(Y_stuck[index])

        # Finding the lowest point in the current interval
        x_min, y_min =  X_in_interval[0], Y_in_interval[0] #Initialization

        for nb in range(len(Y_in_interval)):

            if Y_in_interval[nb] < y_min:
                x_min = X_in_interval[nb]
                y_min = Y_in_interval[nb]

        # The interpolation point for the current interval is the one with the lowest Y coordinate
        X_coordinate_interpolation[point] = (x_min)
        Y_coordinate_interpolation[point] = (y_min)
    
    return X_coordinate_interpolation, Y_coordinate_interpolation


###  Simulation loop ###
# Initialize simulation time
time = 0  
time_list = [0]

while time <= simulation_time:

   # print(f"Time = {time:.2f} s")

    # Add particle in the bottom of the domain

    if time >= Tadding:

        X_add = Lx * np.random.rand(50) # Random X positions
        Y_add = -0.5 * np.random.rand(50) # Random Y positions (Yp in [-0.5, 0])
        U_add = 2 * np.random.rand(50) - 1  # Random X velocities (Up in [-1, 1])
        V_add = 2 + 2 * np.random.rand(50) - 1  # Random Y velocities (Vp in [1, 3])
        stuck_add = np.zeros(50, dtype=bool)

        Xp = np.concatenate((Xp, X_add))
        Yp = np.concatenate((Yp, Y_add))
        Up = np.concatenate((Up, U_add))
        Vp = np.concatenate((Vp, V_add))
        stuck = np.concatenate((stuck, stuck_add))

        Tadding += Tadd

    # Initialization

    Zx_list = np.random.normal(0, 1, len(Xp)) # Normal coefficient for Wiener processus
    Zy_list = np.random.normal(0, 1, len(Xp)) # Normal coefficient for Wiener processus

    # Detect and handle collisions
    coll_pairs = detect_collisions(Xp, Yp, collision_treshold)  # Find colliding particles
    handle_collisions(coll_pairs, Xp, Yp, Up, Vp, stuck)  # Resolve collisions

    # Update positions for moving particle
    for index, element in enumerate(stuck):

        if element == False :
            Xp[index] = (Xp[index] + dt * Up[index] + np.sqrt(2*diffusion_coefficient*dt)*Zx_list[index]) % Lx # Update X positions (periodic boundary conditions)
            Yp[index] = Yp[index] + dt * Vp[index] + np.sqrt(2*diffusion_coefficient*dt)*Zy_list[index] # Update Y positions

        #If particle are stuck, velocities are fixed to 0 in the handle_collisions function

    # Calculation of film thickness V0 (ie, thickness = Y position of the lowest particule)
    """for index, element in enumerate(stuck):
        if element == True:
            y_position_stuck_particle.append(Yp[index])
    
    thickness= Ly - min(y_position_stuck_particle)
    film_thickness.append(thickness)"""

    #Calculation of the interpolations points then used with classical interpolation or B-spline interpolation
    X_coordinate_interpolation, Y_coordinate_interpolation = find_interpolation_points(Xp, Yp, stuck, number_points_interpolation, interval_interpolation_size)

    # Interpolation V1 (ie, using scipy.interpolate.interp1d())
    f = interpolate.interp1d(X_coordinate_interpolation, Y_coordinate_interpolation, fill_value="extrapolate") # Interpolation function based on the points found by interpolation_thickness() 
    X_interpolated = x_abscissa
    Y_interpolated = f(x_abscissa)
    
    # Interpolation V2 (ie, using B-spline)
    """t, c, k = interpolate.splrep(X_coordinate_interpolation, Y_coordinate_interpolation, s = s_parameter_Bspline, k = k_parameter_Bspline) #Definition of the parameter of the B-spline
    spline = interpolate.BSpline(t, c, k) # Interpolation function using a B-spline
    X_interpolated = x_abscissa
    Y_interpolated = spline(x_abscissa)"""

    #Calculation of the tickness = height of the mean point from the interpolated points
    thickness = Ly - sum(Y_interpolated)/len(Y_interpolated)
    if abs(film_thickness[-1]-thickness)/film_thickness[-1] < 0.3 and thickness - film_thickness[-1] > 0:
        film_thickness.append(thickness)
        time_list.append(time)
        
    # Save X, Y positions and materials of interpoilation for the animation every each Tout
    if time >= Toutput: 

        #Interpolation
        X_coordinate_interpolation_animation.append(X_coordinate_interpolation) # Points used for interpolation
        Y_coordinate_interpolation_animation.append(Y_coordinate_interpolation) # Points used for interpolation
        X_interpolated_animation.append(X_interpolated) # Interpolation function
        Y_interpolated_animation.append(Y_interpolated) # Interpolation function

        #Positions and velocities of particles
        Xp_total.append(Xp.copy()) # Update matrix of X positions for each Tout
        Yp_total.append(Yp.copy()) # Update matrix of Y positions for each Tout
        Up_total.append(Up.copy()) # Update matrix of X velocities for each Tout
        Vp_total.append(Vp.copy()) # Update matrix of Y velocities for each Tout
        Toutput += Tout # Update the next output time

    # Total kinetic energy calculation
    if PRINT_KINETIC_ENERGY:

        kinetic_energy = 0
        for index in range(len(Vp)):

            kinetic_energy += np.sqrt(Up[index]**2 + Vp[index]**2)

        print(f"Énergie cinétique totale = {kinetic_energy:.2f} J")
    #Update time 
    time += dt  # Advance simulation time
    #print("---------------------")


### Other calculation ###

# Film growth rate

time_list_tendance = [time_list[0]]
film_thickness_tendance = [film_thickness[0]]

next_point = 0.05

for index in range(len(time_list)):

    if time_list[index] > next_point:

        time_list_tendance.append(time_list[index])
        film_thickness_tendance.append(film_thickness[index])
        next_point += 0.05


modified_film_thickness = film_thickness_tendance[::-1]
modified_film_thickness.append(film_thickness_tendance[0])
modified_film_thickness = modified_film_thickness[::-1]
modified_film_thickness.append(film_thickness_tendance[-1])

film_growth_rate = []
for index in range(len(film_thickness_tendance)):
    film_growth_rate.append((modified_film_thickness[index+2]-modified_film_thickness[index])/(2*dt))

a, b = np.polyfit(time_list, film_thickness, 1) 
print("Régression linéaire (ax+b) : a =", int(a*1000)/1000, " & b =", int(b*1000)/1000)
### Results ###

## Positions of particle ##

X_final, Y_final = Xp_total[-1], Yp_total[-1]
X_saved, Y_saved = [], [] #We only save particle which are stuck

for index in range(len(X_final)):

    if stuck[index] == True:
        X_saved.append(X_final[index])
        Y_saved.append(Y_final[index])

if SAVE_POSITION:

    file_to_write = open("CNRS-EM2C-MODELE-FILM-NANO-PARTICULE/CODE_SOURCE/CSV/film_position_particle_P="+str(int(stick_probability*100)/100)+"_D="+str(diffusion_coefficient)+".csv", 'w')
    writer = csv.writer(file_to_write, delimiter = ",")

    for index in range(len(X_saved)):
        writer.writerow([str(X_saved[index]), str(Y_saved[index])])

    file_to_write.close()   
    print("Positions sauvegardées")

## Animation ##

print("Création de l'animation")

# Create the figure
fig1, ax1 = plt.subplots()
fig1.canvas.manager.set_window_title("animation")
scat1 = ax1.scatter([], [])  # Scatter object for particules
scat2 = ax1.scatter([], []) # Scatter object for the interpolated points
scat3, = ax1.plot([], []) # Plot object for the interpolation function
ax1.set_xlim(0, Lx)  # X limit
ax1.set_ylim(Ly - 1, Ly - collision_treshold/2)  # Y limit
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.tick_params(axis = 'both')

# Update function
def update(frame):
    positions = np.column_stack((Xp_total[frame], Yp_total[frame])) # X and Y positions for the good frame
    scat1.set_offsets(positions)  # Update positions in the scatter plot
    scat1.set_array(0.5 * (Up_total[frame]**2 + Vp_total[frame]**2))
    scat1.set_sizes([15] * len(positions))
    
    interpolation = np.column_stack((X_coordinate_interpolation_animation[frame], Y_coordinate_interpolation_animation[frame]))
    scat2.set_offsets(interpolation)
    scat2.set_color("red")
    scat2.set_sizes([7] * len(interpolation))

    scat3.set_data(X_interpolated_animation[frame], Y_interpolated_animation[frame])
    scat3.set_color("red")

    current_time = frame * Tout  # Calculation of time (frame × interval)
    ax1.set_title(f"Particules animation (time = {current_time:.2f}s)")  # Dynamical title
    return scat1,

# Create the animation

ani = FuncAnimation(fig1, update, frames=int(simulation_time/Tout), interval = 10)

if SAVE_ANIMATION:

    print("Enregistrement de l'animation")
    ani.save('film_collision.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    print("Animation sauvegardée")

### Plot ###

# Plot for the film thickness as a function of time
fig2, ax2 = plt.subplots()
fig2.canvas.manager.set_window_title("film_thickness_P="+str(int(stick_probability*100)/100)+"_D="+str(diffusion_coefficient))
ax2.plot(time_list, film_thickness, color = "black")
#ax2.plot(time_list_tendance, film_thickness_tendance, color = "red", marker = "o")
#ax2.plot(time_list, a*np.array(time_list) + b, color = "red")
ax2.set_xlabel("Time (s)", fontsize = 25)
ax2.set_ylabel("Thickness (µm)", fontsize = 25)
ax2.tick_params(axis = 'both', labelsize = 20)
ax2.grid()
ax2.set_xlim(xmin=0)
ax2.set_ylim(ymin=film_thickness[0])
ax2.set_title("Evolution of film thickness as a function of time", fontsize = 30)

# Plot for the film growth rate as a function of time
"""fig3, ax3 = plt.subplots()
fig3.canvas.manager.set_window_title("film_growth_rate")
ax3.plot(time_list_tendance, film_growth_rate, color = "black")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Growth rate (µm/s)")
ax3.grid()
ax3.set_title("Evolution of film growth rate as a function of time")"""

fig4, ax4 = plt.subplots()
fig4.canvas.manager.set_window_title("film_positions_P="+str(int(stick_probability*100)/100)+"_D="+str(diffusion_coefficient))
ax4.scatter(X_saved, Y_saved, s = 25, marker = "o", color = "black")
ax4.set_xlim(0, Lx)  # X limit
ax4.set_ylim(Ly - 1, Ly - collision_treshold/2)  # Y limit
ax4.set_xlabel("X (µm)", fontsize = 25)
ax4.set_ylabel("Y (µm)", fontsize = 25)
ax4.tick_params(axis = 'both', labelsize = 20)
ax4.set_title("Film after " + str(simulation_time) + "s of simulation", fontsize = 30)



### Runtime ###
Calculation.runtime_program(beginning_date_and_time)

plt.show() 