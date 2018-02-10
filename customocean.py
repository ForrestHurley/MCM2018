import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.interpolate import LinearNDInterpolator
import mcm_utils

class waves:

    def __init__(self):
        self.wave=self.summedTrochoids
        self.waveNorms=self.summedTrochoidNorms
        self.waveParams=([(0,1,0.6),(.57,0.6,0.5),(1,1.2,0.3)],)
        self.resolution=20
        self.waverange=(0,10)
        self.__calculated_wave=None
        self.__calculated_norms=None

        self.__recalculate_norm_interp=True

        self.recalculate_surface=True

        self.normal_smoothing_factor=0.1

    def __calculate_wave_norm(self):
        x = np.linspace(*self.waverange,self.resolution)
        y = np.linspace(*self.waverange,self.resolution)

        x,y = np.meshgrid(x,y)
      
        t = [x.flatten(),y.flatten()]

        self.__calculated_wave = self.getWave(t).reshape(3,*x.shape)
        self.__calculated_norms = self.getWaveNorms(t).reshape(5,*x.shape)

        self.__recalculate_norm_interp=True

        self.__recalculate_surface = False

    @property
    def norms_interpolator(self):
        if self.__recalculate_norm_interp:
            reshaped_norms = self.calculated_norms.reshape(5,-1)
            self.__norms_interpolator = LinearNDInterpolator(reshaped_norms[:2].T,reshaped_norms[2:].T,fill_value=0)
        return self.__norms_interpolator

    @property
    def calculated_wave(self):
        if self.recalculate_surface or _calculated_wave is None:
            self.__calculate_wave_norm()
        return self.__calculated_wave
    
    @property
    def calculated_norms(self):
        if self.recalculate_surface or _calculated_wave is None:
            self.__calculate_wave_norm()
        return self.__calculated_norms

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
        norms=self.extrude(self.trochoidUnextrudedNormals,t,theta,tBoundFunc=self.trochoid,funcParams=(R,d,theta),tBoundFuncArgs=(R,d))
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
        return new_norms

    def numericNormal(self,t):
        gradient = np.gradient(self.wave(t),t)
        gradient.append(np.negative(np.ones(gradient[0].shape)))
        return gradient

    def getWave(self,t):
        return self.wave(t,*self.waveParams)

    def smooth_normals(self,initial_norms):
        initial_norms[2:4] = self.normal_smoothing_factor*initial_norms[2:4]
        return initial_norms

    def getWaveNorms(self,t):
        initial_norms = self.waveNorms(t,*self.waveParams)
        new_norms = self.smooth_normals(initial_norms)
        return new_norms

    def precalculatedWaveAndNorms(self,transpose=True):
        if transpose:
            return self.calculated_wave.reshape(3,-1).T, self.calculated_norms.reshape(5,-1).T

        return self.calculated_wave, self.calculated_norms

    def randomPointsFromVectors(self,vectors): #uses nx3 array for the vectors
        points, norms = self.precalculatedWaveAndNorms(transpose=False)

        random_locs = np.random.uniform(*self.waverange,(vectors.shape[0],2))

        return random_locs
        #e1_list, e2_list = mcm_utils.orthogonalsFromNormals(vectors)

        #projected_random = [np.sum(e1_list*random_locs,axis=1),np.sum(e2_list*random_locs,axis=1)]

        #return np.array(projected_random).T

    def getNormalAtPoints(self,points):
        return self.norms_interpolator(points)

    def getRandomNormals(self,vectors):
        points = self.randomPointsFromVectors(vectors)

        norms = self.getNormalAtPoints(points)

        return norms

    def plot_waves(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a5a98')
        fig.patch.set_color('#1a5a98')

        t,norms = self.precalculatedWaveAndNorms(transpose=False)
        #print(norms.shape)

        colors = np.sqrt(np.sum(np.square(norms[2:4]),axis=0))
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
    vector_list = np.array([[1,0,0],[0,0,-1],[-0.1,0.1,0.5],[1,2,3],[0,1,0]])
    print(wave_instance.getRandomNormals(vector_list))
    wave_instance.plot_waves()
