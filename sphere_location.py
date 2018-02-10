import mcm_utils
import numpy as np
import math

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
    def __init__(self,data, radius=200):
        self.radius=radius
        self.shape=data.shape
        self.latitude=data[0]
        self.longitude=data[1]
        self.values=data[2]
        if self.shape[-2]>3:
            self.auxilary=data[3:]
        
        self.lambert_data=convert_lambert(self)
        self.lambert_x=self.lambert_data[0]
        self.lambert_y=self.lambert_data[1]

    
        self.function_interpolator=scipy.interpolate.CloughTocher2DInterpolator(np.array([self.latitude,self.longitude]).T,self.values)

        new_data=np.array([data[0],data[1],data[2]])
        
        structured=np.core.record.fromarrays(new_data,names='col1,col2,col3',formats='f8,f8,f8')
        latsort=np.sort(structured,order='col1')               
        longsort=np.sort(structured,order='col2')

        phi=deg2rad(longsort[1])
        theta=deg2rad(latsort[0])
    
        partial_phi=np.divide(np.diff(longsort[2]),np.diff(phi))
        partial_theta=np.divide(np.diff(latsort[1]),np.diff(theta))

        phi=mcm_utils.add_convolve(phi)/2
        theta=mcm_utils.add_convolve(theta)/2

        theta_hat=np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)]).T
        phi_hat=np.array([-sin(phi),cos(phi),0]).T

        gradients=np.sum(partial_theta[:,np.newaxis()]*theta_hat,np.divide(1,np.sin(theta))[:,np.newaxis()]*partial_phi[:,np.newaxis()]*phi_hat)
        new_latitude=90-rad2deg(theta)
        new_longitude=rad2deg(phi)

        self.gradient_interpolator=scipy.interpolate.CloughTocher2DInterpolator(np.array([new_latitude,new_longitude]).T,gradients)
     
        
    def convert_lambert(self):
        return np.array([self.longitude, np.sin(deg2rad(90-self.latitude))])
        
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
    print('hi')
