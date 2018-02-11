import numpy as np
import sys
import surface
import materials as mat
from mcm_utils import *

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from heatmap import heatmap

class atmosphere:
    def __init__(self,region_segments=30):
        self.number_rays = 1000
        self.ray_start = [0,6371,0]
        self.ray_direction = [1,1,0]
        
        self.inner_radius = 6371
        self.outer_radius = 6671

        self.setup_rays()
        self.setup_surfaces()
        
        self.heatmap=heatmap(segments=region_segments)
        self.verbose = True

        self.logged_data = []

    def setup_rays(self):
        ray_origins = np.tile(np.array([self.ray_start]),(self.number_rays,1))

        ray_directions = np.tile(np.array([self.ray_direction]),(self.number_rays,1))
        ray_directions = self.randomize_rays(ray_directions)

        self.ray_origins = ray_origins
        self.ray_directions = ray_directions

        self.towards_sky = True

        self.iter = 0

    def randomize_rays(self,rays):
        return rays + np.random.uniform(-0.05,0.05,(self.number_rays,3))

    #if you want to print out other things (or write to files), override this function and set verbose to true
    def print_state(self):
        self.logged_data.append(self.ray_origins)

    def setup_surfaces(self):
        water=mat.simpleWater()
        water.water_model.normal_smoothing_factor=0.1
        air=mat.simpleAtmosphere()
        
        self.ground_surface = surface.sphere(material=water,radius=self.inner_radius)
        self.atmos_surface = surface.sphere(material=air,radius=self.outer_radius)

        self.iter = 0

    def iterate_rays(self,surface):
        if self.verbose:
            self.print_state()
        self.ray_origins, self.ray_directions = surface.reflect_rays(self.ray_origins,
            self.ray_directions)
        #also handles attenuation

    def signal_strength(self):
        return np.amax(self.heatmap.intensity*(100/self.number_rays))

    def simulate(self,iterations):
        for i in range(iterations):
            self.iter += 1
            
            if self.towards_sky:
                self.heatmap.update_regions(self.ray_origins)
                self.iterate_rays(self.atmos_surface)
            else:
                self.iterate_rays(self.ground_surface)

            self.towards_sky = not self.towards_sky
            #self.heatmap.visualize_intensities()
        #np.savetxt('heatmap.csv',result,delimiter=',')
    def draw_sphere(self,ax,radius=200):
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*radius
        y = np.sin(u)*np.sin(v)*radius
        z = np.cos(v)*radius
        ax.plot_surface(x, y, z, color="r")


    def draw_from_log(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.draw_sphere(ax,self.inner_radius)
        data = zip(*self.logged_data)
        data = [zip(*val) for val in data]
        [ax.plot(*vals,color='b',alpha=4/len(data)) for vals in data]

        plt.show()

if __name__=="__main__":
    world = atmosphere(region_segments=60)
    world.simulate(1)
    intensity_final=world.heatmap.get_physical_intensity(14)
    np.save('finalstate',intensity_final)
    world.heatmap.visualize_intensities(mapview=True)
    world.heatmap.visualize_intensities(mapview=False)
