import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.interpolate import LinearNDInterpolator
import mcm_utils

import vtk
from vtk.util.numpy_support import vtk_to_numpy

class precalculated_waves:
    def __init__(self,wave_file):
        self.load_vtk_data(wave_file)

    def load_vtk_data(self,file_name):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()  # Needed because of GetScalarRange

        output = reader.GetOutput()
        # Get the coordinates of nodes in the mesh
        nodes_vtk_array= reader.GetOutput().GetPoints().GetData()

        values_vtk_array = reader.GetOutput().GetPointData().GetArray() 

        self.wavePoints = vtk_to_numpy(nodes_vtk_array)
        self.waveData = vtk_to_numpy(values_vtk_array)

    @property
    def waveGrid(self):
        return self.__waveGrid

    @property
    def waveHeight(self):
        return self.__waveHeight

    @property
    def waveNorms(self):
        return self.__waveNormal

    @property
    def waveMotion(self):
        return self.__waveMotion

    @property
    def waveSalinity(self):
        return self.__waveSalinity

    @property
    def waveTemp(self):
        return self.__waveTemp

    @property
    def fullInterpolator(self):
        if __interpolator is None:
            self.__interpolator = LinearNDInterpolator(self.waveGrid,data,fill_value=0)
        return self.__interpolator

    def wavePropertiesAtLocation(self,location):
        pass

    def generate_normals_from_heights(self):
        gradient = np.gradient(self.waveHeight,self.waveGrid)
        gradient.append(np.negative(np.ones(gradient[0].shape)))
        return gradient
        

    def render_wave(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a5a98')
        fig.patch.set_color('#1a5a98')

        height = self.waveHeight
        norms = self.waveNorms

        colors = np.sqrt(np.sum(np.square(norms[2:4]),axis=0))

        minn, maxx = colors.min(), colors.max()
        normColors = matplotlib.colors.Normalize(minn,maxx)
        m = plt.cm.ScalarMappable(norm=normColors,cmap='Reds')
        m.set_array([])
        fcolors = m.to_rgba(colors)

        ax.plot_surface(t[0],t[1],t[2],facecolors=fcolors)

        minT = np.min(t,axis=(1,2))
        diffT = np.max(np.ptp(t,axis=0))
        
        ax.set_xlim(minT[0],minT[0]+diffT)
        ax.set_ylim(minT[1],minT[1]+diffT)
        ax.set_zlim(minT[2],minT[2]+diffT)
        plt.show()

class statsWave:

    def __init__(self):
        self.scale_factor=0.001
        self.tile_size=100
        self.wind_direction=np.array([[10,10]])
        self.resolution=(256,256)

        self.__recalculate = True

    @property
    def scale_factor(self):
        return self.__scale_factor

    @property
    def tile_size(self):
        return self.__tile_size

    @property
    def wind_direction(self):
        return self.__wind_direction

    @property
    def resolution(self):
        return self.__resolution

    @scale_factor.setter
    def scale_factor(self,scale_factor):
        self.__scale_factor = scale_factor
        self.__recalculate = True

    @tile_size.setter
    def tile_size(self,tile_size):
        self.__tile_size = tile_size
        self.__recalculate = True

    @wind_direction.setter
    def wind_direction(self,wind_direction):
        self.__wind_direction = wind_direction
        self.__recalculate = True

    @resolution.setter
    def resolution(self,resolution):
        self.__resolution = resolution
        self.__recalculate = True

    @property
    def norms_interpolator(self):
        if self.__recalculate:
            reshaped_norms = self.wave_normals.reshape(5,-1)
            self.__norms_interpolator = LinearNDInterpolator(reshaped_norms[:2].T,reshaped_norms[2:].T,fill_value=0)
        return self.__norms_interpolator

    def to_world(self,k):
        k = 2*k - self.resolution
        k = k *np.pi / self.tile_size
        return k

    def phillips(self,k):
        k = self.to_world(k)

        squared = mcm_utils.dot(k,k)

        wind_speed_sqr = mcm_utils.dot(self.wind_direction,self.wind_direction)
        max_wave = wind_speed_sqr/9.8
        norm = mcm_utils.normalize(k)

        dot = mcm_utils.dot(norm,mcm_utils.normalize(self.wind_direction))

        result = self.scale_factor*np.exp(-1/(squared*(max_wave**2)))/(squared**2)*(dot**6)

        damping = 0.001
        ld2 = (max_wave * damping)**2
        result = result * np.exp(-squared*ld2)

        null_k = np.where(squared < 0.00001)

        result[null_k] = 0

        return result

    def getNormalAtPoints(self,points):
        return self.norms_interpolator(points)

    def fourier_amplitude(self,k):

        phillips_val = self.phillips(k)
        divided = np.sqrt(phillips_val/2)

        real = np.random.normal(size=k.shape[0])*divided
        imaginary = 1j*np.random.normal(size=k.shape[0])*divided
        #print(divided)

        #return (0.2+0j)*divided

        return real+imaginary

    def __iterate_tiles(self):
        points = self.grid.reshape(2,self.resolution[0]*self.resolution[1]).T
        flip_points = -self.grid.reshape(2,self.resolution[0]*self.resolution[1]).T

        amplitudes = self.fourier_amplitude(points)
        flip_amplitudes = self.fourier_amplitude(flip_points)

        amplitudes += np.conj(flip_amplitudes)

        amplitudes = amplitudes.reshape(self.resolution)

        waves = np.fft.fft2(amplitudes)

        waves[::2,::2] = -waves[::2,::2]
        waves[1::2,1::2] = -waves[1::2,1::2]

        return np.real(waves)
  
    def make_grid(self,resolution):
        x,y = np.arange(self.resolution[0]),np.arange(self.resolution[1])
        points = np.array(np.meshgrid(x,y))
        return points

    @property
    def grid(self):
        if not self.__recalculate:
            try:
                return self.__grid
            except AttributeError:
                self.__grid = self.make_grid(self.resolution)
        else:
            self.__grid = self.make_grid(self.resolution)
        return self.__grid

    @property
    def world_grid(self):
        return np.transpose(self.to_world(np.transpose(self.grid,(1,2,0))),(2,0,1))

    @property
    def true_world_grid(self):
        return self.grid*self.tile_size/self.resolution[0]
    @property
    def wave_surface(self):
        if not self.__recalculate:
            try:
                return self.__wave_surface
            except AttributeError:
                self.__wave_surface = self.__iterate_tiles()
        else:
            self.__wave_surface = self.__iterate_tiles()
        return self.__wave_surface
    
    def numericNormal(self):
            surface = np.array([*self.world_grid,self.wave_surface])
            gradient = np.gradient(surface)
            gradient = [gradient[1][2],gradient[2][2],-1]
            #gradient.append(np.negative(np.ones(gradient[0].shape)))
            return np.array(gradient)

    @property
    def wave_normals(self):
        try:
            return self.__wave_normals
        except AttributeError:
            self.__wave_normals = self.numericNormal()
        return self.__wave_normals

    def getRandomNormals(self,vectors):
        random_locs = np.random.uniform(0,self.resolution,(vectors.shape[0],2))

        norms = self.getNormalAtPoints(random_locs)

        return norms
    def visualize_wave(self,bNormalColor=False,save_name=None):
        waves = self.wave_surface

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#ffffff')
        fig.patch.set_color('#ffffff')

        points = self.true_world_grid
        normals = self.wave_normals

        colors = np.sqrt(np.sum(np.square(normals[:2]),axis=0))
        #[print(val[0]) for val in colors]
        minn, maxx = colors.min(), colors.max()
        normColors = matplotlib.colors.Normalize(minn,maxx)
        m = plt.cm.ScalarMappable(norm=normColors,cmap='Reds')
        m.set_array([])
        fcolors = m.to_rgba(colors)

        if bNormalColor:
            ax.plot_surface(points[0],points[1],waves,facecolors=fcolors,linewidth=0,rstride=1,cstride=1)
        else:
            ax.plot_surface(points[0],points[1],waves,linewidth=0,rstride=1,cstride=1)
        #ax.plot_wireframe(t[0],t[1],t[2], color='white',linewidth=0.5)

        t = [*points,waves]
        minT = np.min(t,axis=(1,2))
        diffT = np.max(np.max(t,axis=(1,2))-minT)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        #ax.axis("off")
        ax.set_xlim(minT[0],minT[0]+diffT)
        ax.set_ylim(minT[1],minT[1]+diffT)
        ax.set_zlim(minT[2],minT[2]+diffT)
        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name,dpi=600)

class waves:

    def __init__(self,wave_energy=4,wave_count=10,max_shape=0.9):
        self.max_shape = max_shape
        self.wave_count = wave_count
        self.wave_energy = wave_energy

        self.waverange=(0,10)

        self.wave=self.summedTrochoids
        self.waveNorms=self.summedTrochoidNorms
        self.velocity = self.summedTrochoidVels
        #self.waveParams=([(0,1,0.6),(.57,0.6,0.5),(1,1.2,0.3)],)
        self.set_random_wave_params()
        
        self.resolution=128
        self.__calculated_wave=None
        self.__calculated_norms=None

        self.__recalculate_norm_interp=True

        self.recalculate_surface=True

        self.normal_smoothing_factor=0.1

    def set_random_wave_params(self):
        randoms = np.random.dirichlet((1,)*self.wave_count,1)
        energies = randoms*self.wave_energy
        amplitude = np.sqrt(energies)[0]

        wavelength = np.random.uniform(1/self.max_shape,2,self.wave_count)*amplitude

        offset = np.random.uniform(self.waverange[0],self.waverange[1],(self.wave_count,2))

        params = [np.random.uniform(0,3.14159,self.wave_count),wavelength,amplitude,offset]
        params = list(zip(*params))
        self.waveParams = (params,)

    def __calculate_wave_norm(self):
        x = np.linspace(*self.waverange,self.resolution)
        y = np.linspace(*self.waverange,self.resolution)

        x,y = np.meshgrid(x,y)
      
        t = [x.flatten(),y.flatten()]

        self.__calculated_wave = self.getWave(t).reshape(3,*x.shape)
        self.__calculated_norms = self.getWaveNorms(t).reshape(5,*x.shape)
        self.__calculated_vels = self.getVels(t).reshape(5,*x.shape)

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
        if self.recalculate_surface or __calculated_wave is None:
            self.__calculate_wave_norm()
        return self.__calculated_wave
    
    @property
    def calculated_norms(self):
        if self.recalculate_surface or __calculated_norms is None:
            self.__calculate_wave_norm()
        return self.__calculated_norms

    @property
    def wave(self):
        return self.__wave

    @property
    def waveNorms(self):
        return self.__waveNorms

    @property
    def velocity(self):
        return self.__velocity

    @property
    def calculated_vels(self):
        if self.recalculate_surface or __calculated_vels is None:
            self.__calculate_wave_norm()
        return self.__calculated_vels

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

    @velocity.setter
    def velocity(self,velocity):
        self.__recalculate_surface=True
        self.__velocity = velocity

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
    def waverange(self,waverange):
        self.__recalculate_surface=True
        self.__waverange = waverange

    def trochoid(self,t=np.array([0]),R=0.3,d=1,offset=[0,0]):
        x = R*t-d*np.sin(t)+offset[0]
        y = d*np.cos(t)+offset[1]
        return x, y


    def trochoidNormal(self,t=np.array([0]),R=0.3,d=1,offset=[0,0]):
        normals = np.clip(np.broadcast_arrays(np.nan_to_num(d*np.sin(t)/(d*np.cos(t)-R)),1),-6,6)
        return normals

    def trochoidUnextrudedVels(self,t=np.array([0]),R=0.3,d=1,angle=0,offset=[0,0]):
        initial = self.trochoidNormal(t,R,d,offset)
        initial = mcm_utils.normalize(initial.T).T

        wave_length = 2*np.pi*R
        phase_speed = np.sqrt(9.8/2/np.pi*wave_length)

        vels = phase_speed*np.array(np.broadcast_arrays(np.cos(angle)*initial[0],
                                                        np.sin(angle)*initial[0],
                                                        initial[1]))
        return t, vels

    def trochoidUnextrudedNormals(self,t=np.array([0]),R=0.3,d=1,angle=0,offset=[0,0]):
        initial = self.trochoidNormal(t,R,d,offset)
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

    def trochoid3D(self,t=np.array([[0],[0]]),theta=0, R=0.5, d=0.5, offset=[0,0]):
        surface = self.extrude(self.trochoid,t,theta,funcParams=(R,d,offset))
        return surface

    def trochoid3DNorms(self,t=np.array([[0],[0]]),theta=0, R=0.5, d=0.5, offset = [0,0]):
        norms=self.extrude(self.trochoidUnextrudedNormals,t,theta,tBoundFunc=self.trochoid,funcParams=(R,d,theta, offset),tBoundFuncArgs=(R, d, offset))
        return norms

    def trochoid3DVels(self,t=np.array([[0],[0]]),theta=0, R=0.5, d=0.5, offset = [0,0]):
        vels = self.extrude(self.trochoidUnextrudedVels,t,theta,tBoundFunc=self.trochoid,funcParams=(R,d,theta,offset),tBoundFuncArgs=(R, d, offset))
        return vels

    def summedTrochoids(self,t=np.array([[0],[0]]),trochoidParameters=[(0,1,1)]):
        #print(*trochoidParameters)
        surfaces = np.array([self.trochoid3D(t,*params) for params in trochoidParameters])
        summed_surface = np.sum(surfaces[:,2],axis=0)
        new_surface = np.array([*surfaces[0][:2],summed_surface])
        return new_surface

    def summedTrochoidNorms(self,t=np.array([[0],[0]]),trochoidParameters=[(0,1,1)]):
        surface_norms = np.array([self.trochoid3DNorms(t,*params) for params in trochoidParameters])
        summed_norms = np.sum(surface_norms[:,2:4],axis=0)
        new_norms = np.array([*surface_norms[0][:2],*summed_norms,surface_norms[0][4]])
        return new_norms

    def summedTrochoidVels(self,t=np.array([[0],[0]]),trochoidParameters=[(0,1,1)]):
        surface_vels = np.array([self.trochoid3DVels(t,*params) for params in trochoidParameters])
        summed_vels = np.sum(surface_vels[:,2:5],axis=0)
        new_vels = np.array([*surface_vels[0][:2],*summed_vels])
        return new_vels

    def numericNormal(self,t):
        gradient = np.gradient(self.wave(t),t)
        gradient.append(np.negative(np.ones(gradient[0].shape)))
        return gradient

    def getWave(self,t):
        return self.wave(t,*self.waveParams)

    def getVels(self,t):
        return self.velocity(t,*self.waveParams)

    def smooth_normals(self,initial_norms):
        initial_norms[2:4] = self.normal_smoothing_factor*initial_norms[2:4]
        return initial_norms

    def getWaveNorms(self,t):
        initial_norms = self.waveNorms(t,*self.waveParams)
        new_norms = self.smooth_normals(initial_norms)
        return new_norms

    def randomPointsFromVectors(self,vectors): #uses nx3 array for the vectors
        #points, norms = self.precalculatedWaveAndNorms(transpose=False)

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

    def plot_waves(self,bNorms=False,save_name=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', facecolor='#ffffff')
        fig.patch.set_color('#1a5a98')

        t,norms,vels = self.calculated_wave,self.calculated_norms,self.calculated_vels
        #print(norms.shape)

        colors = np.sqrt(np.sum(np.square(vels[2:5]),axis=0))
        #[print(val[0]) for val in colors]
        minn, maxx = colors.min(), colors.max()
        normColors = matplotlib.colors.Normalize(minn,maxx)
        m = plt.cm.ScalarMappable(norm=normColors,cmap='Reds')
        m.set_array([])
        fcolors = m.to_rgba(colors)

        if bNorms:
            ax.plot_surface(t[0],t[1],t[2],facecolors=fcolors,rstride=1,cstride=1)
        else:
            ax.plot_surface(t[0],t[1],t[2],rstride=1,cstride=1)
        #ax.plot_wireframe(t[0],t[1],t[2], color='white',linewidth=0.5)

        minT = np.min(t,axis=(1,2))
        diffT = np.max(np.max(t,axis=(1,2))-minT)
       
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        #ax.axis("off")
        ax.set_xlim(minT[0],minT[0]+diffT)
        ax.set_ylim(minT[1],minT[1]+diffT)
        ax.set_zlim(minT[2],minT[2]+diffT)
        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name,dpi=600)

if __name__ == "__main__":
    #vector_list = np.array([[1,0,0],[0,0,-1],[-0.1,0.1,0.5],[1,2,3],[0,1,0]])
    #print(wave_instance.getRandomNormals(vector_list))
    #wave_instance.plot_waves()
    new_wave = statsWave()
    #print(new_wave.fourier_amplitude(np.array([[63,63]])))
    #print(new_wave.phillips(np.array([[-1,-1]])))
    for i in range(0,8):
        #for j in range(1,5):
        new_wave.wind_direction = np.array([[5*i+0.5,5*i+0.5]])
        print(new_wave.wind_direction)
            #print(0.01+0.05*i,2*j)
            #new_wave = waves(wave_energy=(0.01+0.05*i),wave_count=2*j)
            #new_wave.plot_waves(save_name="trochoid_energy_"+str(0.01+0.05*i)+"_count_" + str(2*j) +".pdf")
        new_wave.visualize_wave(save_name="wind_speed_" + str(np.sqrt(2*(5*i+0.5)**2)) + ".pdf")
