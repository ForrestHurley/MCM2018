import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scipy.optimize

class waves:

    def __init__(self):
        self.wave=self.summedTrochoids
        self.waveNorms=self.summedTrochoidNorms
        self.waveParams=([(1,2,0.3),(0,1,0.6),(0.5,1,0.4)],)
        self.resolution=20
        self.waverange=((0,10),(0,10))
        self.calculated_wave=None
        self.calculated_norms=None

        self.recalculate_surface=True

    @property
    def wave(self):
        return self.__wave

    @property
    def waveNorms(self):
        return self.__waveNorms

    @property
    def waveParams(self):
        return self.__waveParams

    @property
    def resolution(self):
        return self.__resolution

    @property
    def waverange(self):
        return self.__waverange

    @wave.setter
    def wave(self,wave):
        self.__recalculate_surface=True
        self.__wave = wave

    @waveNorms.setter
    def waveNorms(self,waveNorms):
        self.__recalculate_surface=True
        self.__waveNorms = waveNorms

    @waveParams.setter
    def waveParams(self,waveParams):
        self.__recalculate_surface=True
        self.__waveParams = waveParams

    @resolution.setter
    def resolution(self,resolution):
        self.__recalculate_surface=True
        self.__resolution = resolution

    @waverange.setter
    def waverange(self,waveRange):
        self.__recalculate_surface=True
        self.__waverange = waveRange

    def trochoid(self,t=np.array([0]),R=0.3,d=1):
        x = R*t-d*np.sin(t)
        y = R + d*np.cos(t)
        return x, y

    def trochoidNormal(self,t=np.array([0]),R=0.3,d=1):
        normals = np.clip(np.broadcast_arrays(np.nan_to_num(d*np.sin(t)/(d*np.cos(t)-R)),1),-6,6)
        return normals

    def trochoidUnextrudedNormals(self,t=np.array([0]),R=0.3,d=1,angle=0):
        initial = self.trochoidNormal(t,R,d)
        normals = np.array(np.broadcast_arrays(np.cos(angle)*initial[0],np.sin(angle)*initial[0],-1))
        return t,normals

    def getXBounds(self,points,func,funcParams):
        xBounds = np.array([np.amin(points),np.amax(points)])
        x1,x2 = scipy.optimize.broyden1(lambda x: func(x,*funcParams)[0]-xBounds,xBounds,f_tol=1e-2)
        return x1, x2

    def interpolateXYFunc(self,points,func,tBounds,funcParams=(),actualLocs=None):
        interpLocs = np.linspace(tBounds[0],tBounds[1],self.resolution)
        interpPoints = np.array(func(interpLocs,*funcParams))
        if actualLocs is not None:
            interpPoints[0]=actualLocs
        if len(interpPoints[1].shape)>1:
            interpYs = np.array([np.interp(points,interpPoints[0],interpDim) for interpDim in interpPoints[1]])
        else:
            interpYs = np.interp(points,*interpPoints)

        return interpYs

    def extrude(self,func,points,angle=0,tBoundFunc=None,funcParams=(),tBoundFuncArgs=()):
        r = np.sqrt(np.sum(np.square(points),axis=0))
        pointTheta = angle + np.arctan2(points[1],points[0])
        l = r*np.sin(pointTheta)

        actLocs = None
        if tBoundFunc == None:
            tBounds = self.getXBounds(l,func,funcParams)
        else:
            tBounds = self.getXBounds(l,tBoundFunc,tBoundFuncArgs)
            actLocs = np.linspace(l.min(),l.max(),self.resolution)

        newZ = self.interpolateXYFunc(l,func,tBounds,funcParams,actualLocs=actLocs)
        if len(newZ.shape) > 1:
            newPoints = np.array([*points,*newZ])
        else:
            newPoints = np.array([*points,newZ])

        return newPoints

    def trochoid3D(self,t=np.array([[0],[0]]),theta=0, R=0.5, d=0.5):
        surface = self.extrude(self.trochoid,t,theta,funcParams=(R,d))
        return surface

    def trochoid3DNorms(self,t=np.array([[0],[0]]),theta=0, R=0.5, d=0.5):
        norms=self.extrude(self.trochoidUnextrudedNormals,t,theta,tBoundFunc=self.trochoid,funcParams=(R,d),tBoundFuncArgs=(R,d))
        #print(norms)
        return norms

    def summedTrochoids(self,t=np.array([[0],[0]]),trochoidParameters=[(0,1,1)]):
        surfaces = np.array([self.trochoid3D(t,*params) for params in trochoidParameters])
        summed_surface = np.sum(surfaces[:,2],axis=0)
        new_surface = np.array([*surfaces[0][:2],summed_surface])
        return new_surface

    def summedTrochoidNorms(self,t=np.array([[0],[0]]),trochoidParameters=[(0,1,1)]):
        surface_norms = np.array([self.trochoid3DNorms(t,*params) for params in trochoidParameters])
        summed_norms = np.sum(surface_norms[:,2:4],axis=0)
        new_norms = np.array([*surface_norms[0][:2],*summed_norms,surface_norms[0][4]])
        print(new_norms.shape)
        print(surface_norms[0][4].shape)
        return new_norms

    def numericNormal(self,t):
        gradient = np.gradient(self.wave(t),t)
        gradient.append(np.negative(np.ones(gradient[0].shape)))
        return gradient

    def getWave(self,t):
        return self.wave(t,*self.waveParams)

    def getWaveNorms(self,t):
        return self.waveNorms(t,*self.waveParams)

    def precalculatedWaveAndNorms(self,transpose=True):
        if self.calculated_wave == None or self.calculated_norms == None or self.__recalculate_surface:
            x = np.linspace(*self.waverange[0],self.resolution)
            y = np.linspace(*self.waverange[1],self.resolution)

            x,y = np.meshgrid(x,y)
          
            t = [x.flatten(),y.flatten()]

            self.calculated_wave = self.getWave(t).reshape(3,*x.shape)
            self.calculated_norms = self.getWaveNorms(t).reshape(5,*x.shape)

            self.__recalculate_surface = False

        if transpose:
            return self.calculated_wave.reshape(3,-1).T, self.calculated_norms.reshape(5,-1).T

        return self.calculated_wave, self.calculated_norms

    def randomPointsFromVector(self,vector):
        points, norms = self.precalculatedWaveAndNorms(transpose=False)

        projected

    def plot_waves(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a5a98')
        fig.patch.set_color('#1a5a98')

        t,norms = self.precalculatedWaveAndNorms(transpose=False)
        #print(norms.shape)

        colors = np.sqrt(np.sum(np.square(norms[2:3]),axis=0))
        #[print(val[0]) for val in colors]
        minn, maxx = colors.min(), colors.max()
        normColors = matplotlib.colors.Normalize(minn,maxx)
        m = plt.cm.ScalarMappable(norm=normColors,cmap='Reds')
        m.set_array([])
        fcolors = m.to_rgba(colors)

        ax.plot_surface(t[0],t[1],t[2],facecolors=fcolors)
        #ax.plot_wireframe(t[0],t[1],t[2], color='white',linewidth=0.5)

        minT = np.min(t,axis=(1,2))
        diffT = np.max(np.ptp(t,axis=0))
        
        #ax.axis("off")
        ax.set_xlim(minT[0],minT[0]+diffT)
        ax.set_ylim(minT[1],minT[1]+diffT)
        ax.set_zlim(minT[2],minT[2]+diffT)
        plt.show()

if __name__ == "__main__":
    wave_instance = waves()
    wave_instance.plot_waves()
