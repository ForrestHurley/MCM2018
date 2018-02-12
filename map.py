from matplotlib.lines import Line2D   
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import math

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='cea',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,91.,30.))
#m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua')
var=np.linspace(-180,180,num=30,endpoint=False)
final_x=np.concatenate((var,var))

ysample1=np.full(30,-90)
ysample2=np.full(30,90)

intensity=np.load('finalstate.npy')
n=50
for i in range(n):
    value=-180+(360/n)*i
    final_y=np.concatenate((ysample1,ysample2))
    linesx=[value,value]
    linesy=[90,-90]
    x,y=m(linesx,linesy)
    m.plot(x,y,marker=None,color='k')

for k in range(30):
    value=-1+(2/n)*k
    converted=math.asin(value)*(180/math.pi)
    final_y=np.concatenate((ysample1,ysample2))
    linesx=[-180,180]
    linesy=[converted,converted]
    x,y=m(linesx,linesy)
    m.plot(x,y,marker=None,color='k')

plt.title("Cylindrical Equal-Area Projection")
plt.show()
