import numpy as np


class reflection:
    def __init__(self):
        self.reflect_function=mirror
        self.surfaceparams=(,)
        self.normalfunc = lambda x: return np.broadcast([0,0,1],(x.shape[0],3))
        self.attenfunc = lambda x: return 1
        
    def mirror(normal,rays): #takes two arrays and calculates the reflected rays based on the normals 3xn
        #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
        return rays - 2*np.sum(rays*normal,axis=0)/np.sum(normal*normal,axis=0)*normal

    def attenuated_position_reflections(self,rays,points):
        normals = normalfunc(points)
        keep = self.attenfunc(rays,normals)
        return self.reflect_function(normals[keep],rays[keep],*self.surfaceparams)

