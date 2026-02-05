# transform the code in classes
from dataclasses import dataclass
import numpy as np
@dataclass
class Particle:
    def __init__(self, index, mass, radius, position, velocity, show, aggregate_set):
        self.index = index #int
        self.mass = mass #picogram 10**-12
        self.radius = radius #in micrometer 
        self.position = position # 2d array
        self.velocity = velocity # 2d 
        self.show = show #use this boolean to know if the particle is displayed
        self.aggregate_set = aggregate_set #set of aggregated particles
    
    def __str__(self) -> str:
        return (f"Particle {self.index}: position = {self.position}, "
            f"velocity = {self.velocity}, radius = {self.radius}, mass = {self.mass}"
        )

p1 = Particle(0, 10, 0.1, [0,1], [2,3], False, set())
print(p1)