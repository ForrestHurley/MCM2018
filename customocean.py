import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


class waves:

    def __init__(self):
        self.wave=trochoid
        self.waveNorms=trochoidNormal
        self.waveParams=(1,0.5)

    def trochoid(t=np.array([0]),R=1,d=1):
        x = R*t-d*np.sin(t)
        y = R - d*np.cos(t)
        return x, y

    def trochoidNormal(t=np.array([0]),R=1,d=1):
        return [d*np.sin(t)/(d*np.cos(t)-R),1]

    def extrude(func,points,angle,resolution,*args):
        r = np.sum(np.square(points),axis=0)
        pointTheta = angle - np.arctan2(points[1],points[0])
        l = r*np.sin(pointTheta)

        interpPoints = np.arange(np.amin(l),np.amax(l),resolution)
        newZ = np.interp(l,interpPoints,func(l,*args))
        points.append(newZ)

        return points

    def trochoid3D(t=np.array([[0],[0]]),R=1,d=1,theta=0,resolution=0.1):
        return = extrude(trochoid,t,theta,resolution,R,d)

    def trochoid3DNorms(t=np.array([[0],[0]]),R=1,d=1,theta=0,resolution=0.1):
        return = extrude(trochoidNormal,t,theta,resolution,R,d)

    def numericNormal(self,t):
        gradient = np.gradient(self.wave(t),t)
        gradient.append(np.negative(np.ones(gradient[0].shape)))
        return gradient

    def getWaveAtTimes(self,t):
        return self.wave(t,*self.waveParams)

    def getWaveNorms(self,t):
        return self.waveNorms(t,*self.waveParams)

    def plotWaves(self,t):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a5a98')
        plt.subplots_adjust(0,0,1,1)
        fig.patch.set_color('#1a5a98')

        t = getWaveAtTimes(self,t)

        ax.plot_surface(t[0],t[1],t[2])
        ax.plot_wireframe(t[0],t[1],t[2], color='white',linewidth=0.5)

        minT = np.min(t,axis=1)
        maxT = np.max(t,axis=1)
        
        ax.axis("off")
        ax.set_xlim(minT[0],maxT[0])
        ax.set_ylim(minT[1],maxT[1])
        ax.set_zlim(minT[2],maxT[2])
        plt.show()

if __name__ == "__main__":
	

