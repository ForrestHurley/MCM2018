import numpy as np

def mirror(normal,rays):
    #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    return rays - 2*np.sum(rays*normal,axis=1)/np.sum(normal*normal,axis=1)*normal

class reflection:
    def __init__(self):
        self.reflect_function=mirror
        self.waterparams=(,)
        
    def calc_reflection(self,rays)
        return self.reflect_function(normals,rays,*self.waterparams)

        
