import mcm_utils
import numpy as np
import math
import phys_utils
import scipy
from mcm_utils import deg2rad, rad2deg
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt

class sphere_coordinates:
    def __init__(self,regions=30):
        self.segments = 30
        self.build_sphere_mapping()

    def build_sphere_mapping(self):
        longitudes=np.linspace(0,360,num=self.segments,endpoint=False)
        extended_longitudes=np.tile(longitudes,(self.segments,1))
        sin_latitudes=np.linspace(1,-1,num=self.segments,endpoint=False)
        extended_sin_latitudes=np.tile(sin_latitudes,(self.segments,1)).T

        self.upper_right=np.swapaxes(np.array([extended_longitudes,extended_sin_latitudes]),0,2)
        self.rights=extended_longitudes
        self.uppers=extended_sin_latitudes

    def direction_to_region(self,direction_vectors):
        latlongs = mcm_utils.geographic_coordinates(direction_vectors).T
        return self.lat_longs_to_region(latlongs)

    def lat_longs_to_region(self,lat_longs):
        xreg=(lat_longs[1]/360*self.segments).astype(int)
        yreg=((1-np.sin(mcm_utils.deg2rad(lat_longs[0])))*(self.segments/2)).astype(int)
        return [xreg, yreg]

class geocentric_data:
    def convert_lambert(self):
        return np.array([self.longitude, np.sin(deg2rad(np.full(self.latitude.shape,90)-self.latitude))])
    
    def __init__(self,data, radius=200):
        self.radius=radius
        self.shape=data.shape
        print(self.shape)
        self.latitude=data[0]
        self.longitude=data[1]
        self.values=data[2]
        if self.shape[-2]>3:
            self.auxilary=data[3:]
        
        self.lambert_data=self.convert_lambert()
        self.lambert_x=self.lambert_data[0]
        self.lambert_y=self.lambert_data[1]
        
        self.function_interpolator=CloughTocher2DInterpolator(np.array([self.latitude,self.longitude]).T,self.values)

        new_data=np.array([data[0],data[1],data[2]])
        
        latsort=new_data[new_data[:,1].argsort()]
        longsort=new_data[new_data[:,2].argsort()]
        
        phi=longsort[0]
        theta=latsort[0]
    
        phi=np.squeeze(deg2rad(phi))
        theta=np.squeeze(deg2rad(theta))

        indices1 = np.nonzero(np.diff(phi))
        indices2= np.nonzero(np.diff(theta))
        partial_phi=np.squeeze(np.divide(np.take(np.diff(longsort[2]),indices1),np.take(np.diff(phi),indices1)))
        partial_theta=np.squeeze(np.divide(np.take(np.diff(latsort[2]),indices2),np.take(np.diff(theta),indices2)))

        phi=np.squeeze(np.take(mcm_utils.add_convolve(phi)/2,indices1))
        theta=np.squeeze(np.take(mcm_utils.add_convolve(theta)/2,indices2))

        theta_hat=np.squeeze(np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)]).T)
        phi_hat=np.squeeze(np.array([-np.sin(phi),np.cos(phi),np.zeros(phi.shape)]).T)

        inverse_sine=np.divide(1,np.sin(theta))
        term1=partial_theta[:,np.newaxis]*theta_hat
        term2=partial_phi[:,np.newaxis]*phi_hat
        gradients=np.add(term1,inverse_sine[:,np.newaxis]*term2)
        new_latitude=90-rad2deg(theta)
        new_longitude=rad2deg(phi)

#        self.gradient_interpolator=CloughTocher2DInterpolator(np.array([new_latitude,new_longitude]).T,gradients)
     
        
    def visualize_lambert(self, mapview=False):
        plt.pcolormesh(self.lambert_x,self.lambert_y,self.values)
        plt.show()

    def interpolate_data(self, vectors, system='cartesian'):
        if system=='cartesian': 
            return self.function_interpolator(mcm_utils.geographic_coordinates(vectors))
        if system=='spherical':
            latlongs=np.array([90-rad2deg(vectors[2]),rad2deg(vectors[1])]).T
            return self.function_interpolator(latlongs) 
    
    def interpolate_gradient(self,vector, system='cartesian'):
        if system=='cartesian': 
            return self.gradient_interpolator(mcm_utils.geographic_coordinates(vectors))
        if system=='spherical':
            latlongs=np.array([90-rad2deg(vectors[2]),rad2deg(vectors[1])]).T
            return self.gradient_interpolator(latlongs) 
                
if __name__=='__main__':
    frequency=30e6
    theta_i=1
    year=2000
    month=12
    hour=5
    longitudes=np.tile(np.linspace(-180,180,num=20,endpoint=False),(20,1))
    latitudes=np.tile(np.linspace(-90,90,num=20,endpoint=False).T,(20,1)).T
    master=np.swapaxes(np.array([longitudes,latitudes]),0,2)
    narrow=np.reshape(master,(400,2))
    function= np.vectorize(phys_utils.virtual_height)
    #result=function(narrow.T[1],narrow.T[0])
    result=np.full(400,99)
    data=np.array([narrow.T[1],narrow.T[0],result])
    constant=geocentric_data(data)
    constant.visualize_lambert()
