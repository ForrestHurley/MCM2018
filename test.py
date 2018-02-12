from sphere_location import geocentric_data
import numpy as np

latitudes=np.linspace(-90,90,num=99,endpoint=False)
longitudes=np.linspace(-180,180,num=99,endpoint=False)

heights=np.random.rand(99,99)

ground=geocentric_data(latitudes,longitudes,heights)

hi=ground.interpolate_gradient(np.array([[-60,50,100],[100,100,100]]),system='cartesian')
