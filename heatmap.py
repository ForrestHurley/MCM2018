import numpy as np
import mcm_utils

class heatmap:
    def __init__(self,ray_count = 100,total_power=100):
        self.segments=30

        self.reset()
        self.build_sphere_mapping()

        self.initial_ray_count = ray_count
        self.initial_power = total_power

    def build_sphere_mapping(self):
        
        longitudes=np.linspace(0,360,num=self.segments,endpoint=False)
        extended_longitudes=np.tile(longitudes,(self.segments,1))
        sin_latitudes=np.linspace(1,-1,num=self.segments,endpoint=False)
        extended_sin_latitudes=np.tile(sin_latitudes,(self.segments,1)).T

        self.upper_right=np.swapaxes(np.array([extended_longitudes,extended_sin_latitudes]),0,2)
        self.rights=extended_longitudes
        self.uppers=extended_sin_latitudes

    def update_regions(self,intersection_points):
        self.intensity.append(self.accumulate_regions(intersection_points))

    def accumulate_regions(self,intersection_points):
        current_heat = np.zeros((self.segments,self.segments))

        latlongs=mcm_utils.geographic_coordinates(intersection_points).T
        
        xreg=(latlongs[1]/360*self.segments).astype(int)
        yreg=((1-np.sin(mcm_utils.deg2rad(latlongs[0])))*(self.segments/2)).astype(int)
    
        for k in range(xreg.shape[0]):
                current_heat[xreg[k],yreg[k]]+=1

        return current_heat

    def reset(self):
        self.intensity=[np.zeros((self.segments,self.segments))]

    def counts_to_intensity(self,values):
        return values / self.initial_ray_count * self.initial_power

    def get_physical_intensity(self,time_steps=None):
        if time_steps is not None:
            mean_count = np.mean(self.intensity[-time_steps:],axis=0)
            return counts_to_intensity(mean_count)
        else:
            return counts_to_intensity(self.intensity[-1])

    def visualize_intensities(self):
        pass 
