import numpy as np
import mcm_utils
from sphere_location import sphere_coordinates,geocentric_data
import matplotlib.pyplot as plt
import phys_utils as p
import cv2

class heatmap:
    def __init__(self,ray_count = 100,total_power=100,segments=30):
        self.segments=segments

        self.reset()
        self.mapping = sphere_coordinates(self.segments)

        self.initial_ray_count = ray_count
        self.initial_power = total_power

    def update_regions(self,intersection_points):
        self.intensity.append(self.accumulate_regions(intersection_points))

    def accumulate_regions(self,intersection_points):
        current_heat = np.zeros((self.segments,self.segments))

        xreg, yreg = self.mapping.direction_to_region(intersection_points)
    
        for k in range(xreg.shape[0]):
                current_heat[xreg[k],yreg[k]]+=1

        return current_heat

    def reset(self):
        self.intensity=[np.zeros((self.segments,self.segments))]

    def SNR_intensity(self):
        area_region=(p.Rem/self.segments)**2
        power=self.get_physical_intensity()/area_region
        noise=p.k_br*p.temp*p.bandwidth
        
        SNR=power/noise
        return SNR

    def binary_map(self):
        SNR=self.SNR_intensity()
        binary=np.greater(SNR,10).astype(int)

        plt.imshow(binary,cmap=plt.cm.gray)
        np.savetxt('binary.csv',SNR,delimiter=',')
        #ret,labels=cv2.connectedComponents(np.greater(SNR,10).astype(int))
        return 0

    def counts_to_intensity(self,values):
        return values / self.initial_ray_count * self.initial_power

    def get_physical_intensity(self,time_steps=None):
        if time_steps is not None:
            mean_count = np.mean(self.intensity[-time_steps:],axis=0)
            return self.counts_to_intensity(mean_count)
        else:
            return np.sum(self.counts_to_intensity(np.array(self.intensity)),axis=0)

    def visualize_intensities(self,mapview=False):
        heatmap_intensities=geocentric_data(self.mapping.latitudes,self.mapping.longitudes,self.get_physical_intensity())
        heatmap_intensities.visualize_lambert(mapview=mapview)
        
