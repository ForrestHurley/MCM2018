

class material:

    def attenuate(self,ray_direction,material_normal,intersection_location):
        def f(*args):
            return args
        return f

    def normal(self,direction,location):
        return numpy.array([*np.zeros(location.shape[0],2),np.ones(location.shape[0])])

class simpleWater(material):
    from customocean import waves

    def __init__(self,turbulence):
        self.surface = waves()

    def normal(self,direction,location):
        self.surface.getRandomNormals(direction)

class simpleAtmosphere(material):

