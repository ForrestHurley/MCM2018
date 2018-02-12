import mcm_utils
import numpy as np
import math
import scipy
from mcm_utils import deg2rad, rad2deg
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import phys_utils
from mpl_toolkits.basemap import Basemap

class sphere_coordinates:
    def __init__(self,segments=30):
        self.segments = segments
        self.build_sphere_mapping()

    def build_sphere_mapping(self):
        longitudes=np.linspace(0,360,num=self.segments,endpoint=False)
        extended_longitudes=np.tile(longitudes,(self.segments,1))
        sin_latitudes=np.linspace(1,-1,num=self.segments,endpoint=False)
        self.longitudes=np.linspace(-180,180,num=self.segments,endpoint=True)
        self.latitudes=rad2deg(np.arcsin(np.linspace(1,-1,num=self.segments,endpoint=True)))
        extended_sin_latitudes=np.tile(sin_latitudes,(self.segments,1)).T

        self.upper_right=np.swapaxes(np.array([extended_longitudes,extended_sin_latitudes]),0,2)
        self.rights=extended_longitudes
        self.uppers=extended_sin_latitudes

    def direction_to_region(self,direction_vectors):
        latlongs = mcm_utils.geographic_coordinates(direction_vectors).T
        return self.lat_longs_to_region(latlongs)

    def lat_longs_to_region(self,lat_longs):
        xreg=((lat_longs[1]+180)/360*self.segments).astype(int)
        yreg=((np.sin(mcm_utils.deg2rad(90-lat_longs[0])))*(self.segments/2)).astype(int)
        return [yreg, xreg]

class geocentric_data:
    def convert_lambert(self):
        lambert_coordinate=np.sin(deg2rad(self.latitude))
        return np.array([self.longitude, lambert_coordinate])

    #data must be a sorted array of latitudes, longitudes, and then grid mesh of the coordinates.    
    def __init__(self,latitudes, longitudes, data, radius=200):
        self.radius=radius
        self.shape=data.shape
        self.latitude=latitudes
        self.longitude=longitudes
        self.values=data
        
        self.lambert_data=self.convert_lambert()
        self.lambert_x=self.lambert_data[0]
        self.lambert_y=self.lambert_data[1]
        
        coordinates=np.swapaxes(np.array(np.meshgrid(self.latitude,self.longitude)),0,2)        
        interpshape=(coordinates.shape[0]**2,2)
        flatshape=(coordinates.shape[0]**2,)
        cartshape=(coordinates.shape[0]**2,3)
        self.function_interpolator=CloughTocher2DInterpolator(np.reshape(coordinates,interpshape),np.reshape(self.values,flatshape))
        phi=deg2rad(self.longitude)
        theta=deg2rad(90-self.latitude)

        sphere_coordinates=np.meshgrid(theta,phi)
        
        theta_mesh=sphere_coordinates[0]
        phi_mesh=sphere_coordinates[1]
        
        sphere_coordinates=np.swapaxes(sphere_coordinates,0,2)
        
        sphere_gradient=np.gradient(self.values,math.pi/self.latitude.shape[0],(2*math.pi)/self.longitude.shape[0])
        
        partial_theta=np.squeeze(sphere_gradient[0])
        partial_phi=np.squeeze(sphere_gradient[1])

        theta_hat=np.swapaxes(np.array([np.cos(theta_mesh)*np.cos(phi_mesh),np.cos(theta_mesh)*np.sin(phi_mesh),-np.sin(theta_mesh)]),0,2)
        phi_hat=np.swapaxes(np.array([-np.sin(phi_mesh),np.cos(phi_mesh),np.zeros(phi_mesh.shape)]),0,2)
        
        inverse_sine=np.divide(1,np.sin(theta_mesh))
        term1=partial_theta[:,:,np.newaxis]*theta_hat
        term2=partial_phi[:,:,np.newaxis]*phi_hat
        gradients=np.add(term1,inverse_sine[:,:,np.newaxis]*term2)

        self.gradient_interpolator=CloughTocher2DInterpolator(np.reshape(coordinates,interpshape),np.reshape(gradients,cartshape))
        
        
    def visualize_lambert(self, mapview=False,log_scale=True):
        cmap='winter'
        if mapview:
            # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
            # are the lat/lon values of the lower left and upper right corners
            # of the map.
            # resolution = 'c' means use crude resolution coastlines.
            m = Basemap(projection='cea',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
            mapx,mapy=m(self.longitude,self.latitude)
            print(self.values)
            m.pcolormesh(mapx,mapy,self.values,cmap=cmap,
                norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                vmin=self.values.min(), vmax=self.values.max()))
            #m.fillcontinents(color='coral',lake_color='aqua')
            m.drawcoastlines()
            # print(mapx)
            # draw parallels and meridians.
            #m.drawparallels(np.arange(-90.,91.,30.))
            #m.drawmeridians(np.arange(-180.,181.,60.))
            #m.drawmapboundary(fill_color='aqua')
        else:
            plt.pcolormesh(self.lambert_x,self.lambert_y,self.values,cmap=cmap,
                norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                vmin=self.values.min(), vmax=self.values.max()))
        plt.show()

    def interpolate_data(self, vectors, system='cartesian'):
        if system=='cartesian': 
            return self.function_interpolator(mcm_utils.geographic_coordinates(vectors))
        if system=='spherical':
            latlongs=np.array([90-rad2deg(vectors[2]),rad2deg(vectors[1])]).T
            return self.function_interpolator(latlongs) 
        if system=='geographic':
            return self.function_interpolator(vectors)
    
    def interpolate_gradient(self,vector, system='cartesian'):
        if system=='cartesian': 
            return self.gradient_interpolator(mcm_utils.geographic_coordinates(vectors))
        if system=='spherical':
            latlongs=np.array([90-rad2deg(vectors[2]),rad2deg(vectors[1])]).T
            return self.gradient_interpolator(latlongs)
        if system=='geographic':
            return self.gradient_interpolator(vector) 

if __name__=='__main__':
    longitudes=np.linspace(-180,180,num=20,endpoint=False)
    latitudes=np.linspace(-90,90,num=20,endpoint=False)
    #data=np.zeros((20,20))
    #for i in range(latitudes.shape[0]):
    #    for k in range(longitudes.shape[0]):
    #        #print("for latitude",latitudes[i],"and for longitude",longitudes[k])
    #        profile=phys_utils.electron_density_profile(latitudes[i],longitudes[k])
    #        data[i][k]=np.amax(profile)
    #np.save('electron_density',data)
    data=np.load('electron_density.npy')
    constant=geocentric_data(latitudes,longitudes,data)
    maxvector=np.array([20,50])
#    print(constant.interpolate_gradient(maxvector,system='geographic'))
