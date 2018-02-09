

class material:

    def attenuation(direction,location):
        return 0

    def normal(direction,location):
        return numpy.array([*np.zeros(interaction_point.shape[0],2),np.ones(interaction_point.shape[0])])

class simpleWater(material):
    from customocean import waves

    def __init__(self,turbulence):
        self.surface = waves()

    def attuation(direction,location):
        return 0

    def normal(direction,location):
        self.surface.

class simpleAtmosphere(material):

    def attenuation(direction,location):
        return 0
