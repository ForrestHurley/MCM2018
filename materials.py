

class default_mat:
    def __init__():
        pass

    def attenuate(self,ray_direction,material_normal,intersection_location):
        def f(*args):
            return args
        return f

    def normals(self,direction,location):
        return numpy.array([*np.zeros(location.shape[0],2),np.ones(location.shape[0])])

class simpleWater(default_mat):
    from customocean import waves

    def __init__(self,turbulence):
        super().__init__()
        self.surface = waves()

    def normal(self,direction,location):
        self.surface.getRandomNormals(direction)

class simpleAtmosphere(default_mat):
    pass
