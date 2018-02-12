from sphere_location import geocentric_data
import numpy as np

latitudes=np.linspace(-90,90,num=100,endpoint=False)
longitudes=np.linspace(-180,180,num=100,endpoint=False)

heights=np.random.rand(100,100)

ground=geocentric_data(latitudes,longitudes,heights)


