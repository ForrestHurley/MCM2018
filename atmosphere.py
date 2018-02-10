import numpy as np
import sys
from reflection import *
from surface import *
import utils_mcm
import materials as mat

class atmosphere:
    def __init__(self):
        self.number_of_rays = 40
        self.ray_start = [200,0,0]
        self.ray_direction = [1,1,0]
   
        self.inner_radius = 200
        self.outer_radius = 220

        self.setup_rays()
        self.setup_surfaces()

        self.verbose = True

    def setup_rays(self):
        ray_origins = np.tile(np.array([self.ray_start]),(self.number_rays,1))

        ray_directions = np.tile(np.array([self.ray_direction]),(self.number_rays,1))
        ray_directions = randomize_rays(ray_directions)

        self.ray_origins = ray_origins
        self.ray_directions = ray_directions

        self.towards_sky = True

        self.iter = 0

    def randomize_rays(self,rays):
        return rays + np.random.range(-0.05,0.05,(self.number_rays,3))

    #if you want to print out other things (or write to files), override this function and set verbose to true
    def print_state(self):
        print(self.ray_origins)

    def setup_surfaces(self):
        self.ground_surface = sphere(material=mat.simpleWater,radius=self.inner_radius)
        self.atmos_surface = sphere(material=mat.simpleAtmosphere,radius=self.outer_radius)

        self.iter = 0

    def iterate_rays(self,surface):
        self.ray_origins, self.ray_directions = 
            surface.reflect_rays(self.ray_origins,self.ray_directions) #also handles attenuation
        if self.verbose:
            self.print_state()

    def simulate(self,iterations):
        for i in range(iterations):
            self.iter += 1

            if towards_sky:
                self.iterate_rays(self.atmos_surface)
            else:
                self.iterate_rays(self.ground_surface)

            towards_sky = !towards_sky

        
if __name__=="__main__":
    world = atmosphere()
    world.simulate(10)
