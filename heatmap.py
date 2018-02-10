import numpy as np
import mcm_utils

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
        latlongs=mcm_utils.geographic_coordinates(intersection_points).T
        xreg=(latlongs[1]//12).astype(int)
        yreg=((1-np.sin(mcm_utils.deg2rad(latlongs[0])))//(2/30)).astype(int)
        for k in range(xreg.shape[0]):
                self.intensity[xreg[k],yreg[k]]+=1

    def reset(self):
        self.intensity=np.zeros((30,30))

