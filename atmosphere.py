import numpy as np
import sys
import surface
import materials as mat
from mcm_utils import *

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class atmosphere:
    def __init__(self):
        self.number_rays = 1000
        self.ray_start = [200,0,0]
        self.ray_direction = [1,1,0]
   
        self.inner_radius = 200
        self.outer_radius = 220

        self.setup_rays()
        self.setup_surfaces()
        
        self.heatmap=heatmap()
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
        self.ground_surface = surface.sphere(material=mat.simpleWater,radius=self.inner_radius)
        self.atmos_surface = surface.sphere(material=mat.simpleAtmosphere,radius=self.outer_radius)

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
        out=np.zeros((int(iterations/2),30,30))
        for i in range(iterations):
            self.iter += 1
            
            if self.towards_sky:
                self.heatmap.update_regions(self.ray_origins)
                out[int(i/2)]=self.heatmap.intensity
                image=plt.imshow(self.heatmap.intensity)
                self.iterate_rays(self.atmos_surface)
            else:
                self.iterate_rays(self.ground_surface)

            self.towards_sky = not self.towards_sky
        result=np.concatenate(out)
        np.savetxt('heatmap.csv',result,delimiter=',')
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
        data = np.array(self.logged_data)
        data = np.transpose(data,(1,2,0))
        [ax.plot(*vals,color='b',alpha=4/data.shape[0]) for vals in data]

        plt.show()

class heatmap:
    def __init__(self):
        self.intensity=np.zeros((30,30))
        longitudes=np.linspace(0,360,num=30,endpoint=False)
        extended_longitudes=np.tile(longitudes,(30,1))
        sin_latitudes=np.linspace(1,-1,num=30,endpoint=False)
        extended_sin_latitudes=np.tile(sin_latitudes,(30,1)).T
        self.upper_right=np.swapaxes(np.array([extended_longitudes,extended_sin_latitudes]),0,2)
        self.rights=extended_longitudes
        self.uppers=extended_sin_latitudes

    def update_regions(self,intersection_points):
        self.reset()
        latlongs=geographic_coordinates(intersection_points).T
        xreg=(latlongs[1]//12).astype(int)
        yreg=((1-np.sin(deg2rad(latlongs[0])))//(2/30)).astype(int)
        for k in range(xreg.shape[0]):
                self.intensity[xreg[k],yreg[k]]+=1            
        
    def reset(self):
        self.intensity=np.zeros((30,30))    

if __name__=="__main__":
    world = atmosphere()
    world.simulate(30)
    world.draw_from_log()
