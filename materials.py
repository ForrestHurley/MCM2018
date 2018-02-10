import numpy as np
from customocean import waves

class default_mat:
    def __init__(self):
        pass

    def attenuate(self,ray_direction,material_normal,intersection_location):
        def f(*args):
            return args
        return f

    def normals(self,direction,location):
        return np.array([*np.zeros((2,location.shape[0])),np.ones(location.shape[0])])

class simpleWater(default_mat):

    def __init__(self,turbulence=0.1):
        super().__init__()
        self.water_model = waves()

    def normals(self,direction,location):
        return self.water_model.getRandomNormals(direction).T

class simpleAtmosphere(default_mat):
    pass
