#I am wanting to code a mpi code that can send one particule from one proc to another
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD #INIT the mpi
my_rank = comm.Get_rank()
my_size = comm.Get_size()

a = 0.0
b = 1.0
n = 1024
dest = 0
total = -1.0
h = (b - a) / n

local_n = n // my_size
local_a = a + my_rank * local_n * h;
local_b = local_a + local_n * h 
# build local grid and values
x_local = np.linspace(local_a, local_b, local_n + 1 )
y_local = (x_local)**2

# local integral
integral = np.trapezoid(y_local, dx=h)
#add up the integrals
if my_rank == 0:
    total = integral
    for source in range (1,my_size):
        integral = comm.recv(source = source)
        print ("Processor", my_rank, "<-", source, ",", integral, "\n")
        total = total + integral 
else:
    print("Processor", my_rank, "->", dest, ",", integral, "\n")
    comm.send(integral, dest = 0)
    
#print the result
if(my_rank == 0) :
    print("With n = ", n, "Integral from ", a, "to ", b, "is equal to ", total)
    
MPI.Finalize