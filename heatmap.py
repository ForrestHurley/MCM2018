import numpy as np
import mcm_utils
from sphere_location import sphere_coordinates,geocentric_data
import matplotlib.pyplot as plt
import phys_utils as p
import skimage.measure as sm
import math
np.set_printoptions(threshold=np.nan)
class heatmap:
    def __init__(self,origin,ray_count = 100,total_power=100,segments=30):
        self.segments=segments
        self.origin=origin
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

    def SNR_intensity(self,array=None):
        if type(array)==type(None):
            array=self.get_physical_intensity()
        area_region=(p.Rem/self.segments)**2
        power=array/area_region
        noise=p.k_br*p.temp*p.bandwidth
        
        SNR=power/noise
        return SNR

    def binary_map(self,array=None, labeling=True):
        SNR=self.SNR_intensity(array=array)
        binary=np.greater(SNR,10).astype(int)
        #plt.imshow(binary,cmap=plt.cm.gray)
        #plt.show()
        if labeling:
            return sm.label(binary,return_num=True,connectivity=1)
        else:
            return binary

    def metrics(self):
        labels,num_labels=self.binary_map()
        sizes=[]
        num_regions=0
        for k in range(1,num_labels):
            size=np.sum(labels==k)
            sizes.append(size)
            if size>3:
                num_regions+=1
        return num_regions,max(sizes)*(math.pi*6371**2)/self.segments**2

       # distances=[]
       # cartesians=[]
       # for k in range(num_regions+1):
       #     intensities=self.get_physical_intensity(k+1)
       #     #geocentric_data(self.mapping.latitudes,self.mapping.longitudes,self.get_physical_intensity(2)).visualize_lambert()
       #     binary,num=self.binary_map(array=intensities)
       #     indices=np.array(np.nonzero(binary))
       #     center=np.flip(np.mean(indices,axis=1),0)
       #     center_latlong=np.array([[center[0]*(360/self.segments)-180,(180/math.pi)*np.arcsin(center[1])]])
       #     cartesian=mcm_utils.cartesian_coordinates(center_latlong)
       #     if k==0:
       #         cartesians.append(cartesian)
       #         continue
       #     angle=mcm_utils.angle(cartesians[-1],cartesian)/(math.pi*2)
       #     distance=angle*6371
       #     distances.append(distance)
       # 
       # print(num_regions,max(sizes))
       # print(distances)
       # plt.imshow(binary,cmap=plt.cm.gray)
       # plt.show()
       # #ret,labels=cv2.connectedComponents(np.greater(SNR,10).astype(int))

    def counts_to_intensity(self,values):
        return np.array(values) / self.initial_ray_count * self.initial_power

    def get_physical_intensity(self,time_steps=None):
        if time_steps is not None:
            mean_count = self.intensity[time_steps]
            return self.counts_to_intensity(mean_count)
        else:
            return np.sum(self.counts_to_intensity(np.array(self.intensity)),axis=0)

    def visualize_intensities(self,mapview=False,show=False,*args,**vargs):
        heatmap_intensities=geocentric_data(self.mapping.latitudes,self.mapping.longitudes,self.get_physical_intensity())
        return heatmap_intensities.visualize_lambert(mapview=mapview,show=show,*args,**vargs)
        
