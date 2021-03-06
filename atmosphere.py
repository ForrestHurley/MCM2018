import numpy as np
import sys
import surface
from surface import bumpy_sphere
import materials as mat
from mcm_utils import *
import random
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from heatmap import heatmap
from test import ground
import customocean

class atmosphere:
    def __init__(self,ray_count=10,ray_start=[60,30],ray_direction=[20,45],
                    earth_mat=None,atmos_mat=None,earth_surface=None,atmos_surface=None,
                    region_segments=30):
        self.number_rays = ray_count
        self.ray_start = cartesian_coordinates(np.array([ray_start]))[0]
        self.ray_direction = local_frame_to_cartesian(ray_direction[0],ray_direction[1],ray_start[0],ray_start[1])
        self.earth_mat=earth_mat
        self.atmos_mat=atmos_mat
        self.earth_primitive=earth_surface
        self.atmos_primitive=atmos_surface

        self.inner_radius = 6371
        self.outer_radius = 6671

        self.setup_rays()
        self.setup_surfaces()
        
        self.heatmap=heatmap(origin=self.ray_start,segments=region_segments,ray_count=self.number_rays)
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
        if self.earth_mat is None:
            self.earth_mat=mat.simpleWater()
            self.earth_mat.water_model.normal_smoothing_factor=0.151
        if self.atmos_mat is None:
            self.atmos_mat=mat.simpleAtmosphere()
        
        if self.earth_primitive is None:
            self.earth_primitive = surface.sphere
        if self.atmos_primitive is None:
            self.atmos_primitive = surface.sphere

        self.ground_surface = self.earth_primitive(material=self.earth_mat,
            radius=self.inner_radius)
        self.atmos_surface = self.atmos_primitive(material=self.atmos_mat,
            radius=self.outer_radius)

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
            print(i)
            self.iter += 1
            if self.towards_sky:
                self.iterate_rays(self.atmos_surface)
                self.heatmap.update_regions(self.ray_origins)
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
def thing1():
    #ionosphere=surface.ionosphere
    #bumpy=surface.bumpy_sphere
    for i in range(20): #wave energy
        for j in range(10): #Angle
            for k in range(4): #Albedo
                water_model = customocean.waves(wave_energy=i*0.5+0.1,wave_count=20)
                earth_mat = mat.fresnelWater()
                earth_mat.water_model = water_model
                earth_mat.water_model.normal_smoothing_factor=1
                #earth_mat.water_model = customocean.statsWave()
                #earth_mat.water_model.wind_direction = np.array([[0,i*2]])
                atmos_mat = mat.simpleAtmosphere()#mat.physicalAtmosphere()
                atmos_mat.albedo = k*0.15+0.4


                world = atmosphere(ray_count=100000,ray_direction=[j*5,0],region_segments=400, earth_mat=earth_mat,atmos_mat=atmos_mat) #, atmos_surface=ionosphere,earth_surface=bumpy)
                world.simulate(10)
                #world.draw_from_log()
                #np.save('finalstate',intensity_final)
                #data = world.heatmap.visualize_intensities(mapview=True,show=False,cmap="Reds")
                num_skips,region_size=world.heatmap.metrics()        
               
                print("engAngAlbSkpSiz," + str(i*0.5+0.1) + "," + str(j*5) + "," + str(k*0.15+0.4) + "," + str(num_skips) + "," + str(region_size))
                #print("number of skips",num_skips)
                #print("max region size",region_size)
                #data.savefig("heatmap_time_"+str(i*5+5)+".pdf",dpi=600)
            #world.heatmap.visualize_intensities(mapview=False)
if __name__=="__main__":
    #ionosphere=surface.ionosphere
    #bumpy=surface.bumpy_sphere
    earth_mat = mat.simpleWater()#mat.fresnelWater()
    atmos_mat = mat.simpleAtmosphere()#mat.physicalAtmosphere()

    earth_mat.water_model.normal_smoothing_factor=1
    for k in range(10):
        atmos_mat.albedo=np.random.uniform(0.8,1.00001)
        world = atmosphere(ray_count=1000,region_segments=100,earth_surface=bumpy_sphere,earth_mat=earth_mat,atmos_mat=atmos_mat,ray_direction=[np.random.uniform(0,3),np.random.uniform(0,360)])
        world.simulate(30)
            #np.save('finalstate',intensity_final)
#            data = world.heatmap.visualize_intensities(mapview=True,cmap="Reds")
        num_skips,region_size=world.heatmap.metrics()        
        print(num_skips,",",region_size)
    '''
    for i in range(5):
        water_model = customocean.waves(wave_energy=10,wave_count=20)
        earth_mat = mat.fresnelWater()
        earth_mat.water_model = water_model
        earth_mat.water_model.diffusion_factor=0.2*i+0.1
        #earth_mat.water_model = customocean.statsWave()
        #earth_mat.water_model.wind_direction = np.array([[3,3]])
        atmos_mat = mat.simpleAtmosphere()#mat.physicalAtmosphere()
        world = atmosphere(ray_count=100000,region_segments=400, earth_mat=earth_mat,atmos_mat=atmos_mat) #, atmos_surface=ionosphere,earth_surface=bumpy)
        world.simulate(10)
        #world.draw_from_log()
        #np.save('finalstate',intensity_final)
        data = world.heatmap.visualize_intensities(mapview=True,show=False,cmap="Reds")
        num_skips,region_size=world.heatmap.metrics()        
        #print("number of skips",num_skips)
        #print("max region size",region_size)
        data.savefig("heatmap_diffusion_"+str(i*0.2+0.1)+".pdf",dpi=600)
    #world.heatmap.visualize_intensities(mapview=False)
    '''
