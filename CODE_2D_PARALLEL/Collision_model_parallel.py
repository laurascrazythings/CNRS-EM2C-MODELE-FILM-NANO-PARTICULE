#I am now going to try to recode what was done in the source code for collision_model using the mpi method
from mpi4py import MPI
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

simulation_time = 1 # Total simulation time

# Runtime calculation
beginning_date_and_time = dtm.datetime.now()


# Animation saving
SAVE_ANIMATION = False # Constant to control animation saving

# Position saving
SAVE_POSITION = True # Constant to control position saving in a csv

# Print kinetic energy
PRINT_KINETIC_ENERGY = False # Constant to control kinetic energy printing