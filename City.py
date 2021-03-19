import numpy as np, random, operator, matplotlib.pyplot as plt
from math import sqrt

class City:

    def __init__(self, id, x, y):
       self.pos = (x,y)
       self.ID = id

    def distance(self, city):
        xDis = abs(self.pos[0] - city.x)
        yDis = abs(self.pos[1] - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def toString(self):
        return "City at " + str(self.pos)