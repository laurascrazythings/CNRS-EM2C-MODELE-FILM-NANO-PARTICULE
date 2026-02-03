# transform the code in classes
from dataclasses import dataclass
import numpy as np

class Particle:
    def __init__(self, index, mass, radius, position, velocity):
        self.index = index #int
        self.mass = mass #picogram 10**-12
        self.radius = radius #in micrometer 
        self.position = position # 2d array
        self.velocity = velocity # 2d 
    
    def __str__(self) -> str:
        return (f"Particle {self.index}: position = {self.position}, "
            f"velocity = {self.velocity}, radius = {self.radius}, mass = {self.mass}"
        )