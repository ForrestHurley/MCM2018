from mcm_utils import *

class surface:
    def __init__(self,material=None):
        self.material = material
    def intersection_point(ray_origin,ray_direction):
        return ray_origin
    def normal_from_surface(intersection_location):
        return intersection_location
    def 

class sphere(surface):
    def __init(self,material=None,center=[0,0,0],radius=1)
        surface.__init__(material)
        self.center = np.array(center)
        self.radius = radius

    def intersection_point(ray_origin,ray_direction):
        difference = ray_origin - self.center

        a = dot(ray_direction,ray_direction)
        b = dot(ray_direction,difference)
        c = dot(difference,difference) - self.radius * self.radius

        dist = -b + (b**2-a*c)**0.5

        return ray_origin + dist*ray_direction

    def normal_from_surface(intersection_location):
        return intersection_location - center

class plane(surface):
    def __init__(self,material=None,center=[0,0,0],normal=[0,0,1]):
        surface.__init__(material)
        self.center = np.array(center)
        self.normal = np.array(normal)

    def intersection_point(ray_origin,ray_direction):
        difference = ray_origin - self.center

        dot1 = dot(difference,self.normal)
        dot2 = dot(ray_direction,self.normal)

        return self.center + dot1/dot2*ray_direction

    def normal_from_surface(intersection_location):
        return self.normal
