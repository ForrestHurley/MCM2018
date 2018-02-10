import mcm_utils
import numpy as np

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
