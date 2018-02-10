import numpy as np
from customocean import waves
import scipy
import mcm_utils
from phys_utils import *

class default_mat:
    def __init__(self,albedo=0.9):
        self.albedo = albedo

    def attenuate(self, *args):
        return mcm_utils.random_removal(self.albedo, args)

    def normals(self,direction,location):
        return np.array([*np.zeros((2,location.shape[0])),np.ones(location.shape[0])])

class fresnelMaterial(default_mat):
    def __init__(self,index_of_refraction=1.33):
        super().__init__()
        self.ref_index = index_of_refraction

    def attenuate(self, *args):
        direction_estimate = args[0][0]  # I feel like this is wrong, but I needed to get an angle of incidence somehow 
        dir_mag = np.sqrt(direction_estimate.dot(direction_estimate))
        normal_estimate = args[1][0]
        normal_mag = np.sqrt(normal_estimate.dot(normal_estimate))
        theta_i = np.acos(direction_estimate.dot(normal_estimate) / (dir_mag * normal_mag))
        if theta_i > np.pi/2:  # Just wanted to be sure that the angle of incidence was less than 90deg
            theta_i = np.pi - theta_i
        self.albedo = index_reflectance(theta_i, self.ref_index)
        return super(self.albedo).attenuate(args)  # I think this is how it works


class simpleWater(default_mat):
    def __init__(self,turbulence=0.1):
        super().__init__()
        self.water_model = waves()

    def normals(self,direction,location):
        return self.water_model.getRandomNormals(direction).T

    def attenuate(self, *args):
        pass  # I don't know how attenuation is modeled in this class, if at all


class simpleAtmosphere(default_mat):
    pass

class fresnelWater(fresnelMaterial, simpleWater):
    def __init__(self, index_of_refraction=0.5):
        super().__init__(index_of_refraction)
    

class fresnelAtmosphere(simpleAtmosphere):
    def __init__(self, index_of_refraction=0.5):
        super().__init__(index_of_refraction)


class physicalWater(simpleWater):
    pass

class physicalAtmosphere(simpleAtmosphere):
    def __init__(self, time, month):  # Provide time as a military time string for the given location (2:20PM = 1420)
                                    # Provide day as a number and use a function to map to other months
        super().__init__()
        self.time = time

    def attenuate(self, *args):
        # Calculate the concentrations of N_2, O_2, 


class simpleDirt(default_mat):
    pass

class multiMat(default_mat):
    def __init__(self,mat_list=[default_mat()],mat_region=[[0],[[1,0,0]]]):
        self.material_list = mat_list
        self.mat_region = mat_region    

    def build_interpolator(self):
        self.interpolator = scipy.interpolate.NearestNDInterpolator(self.mat_region[1],self.mat_region[0])

    def get_region(self, locations):
        return self.interpolator(locations)

    def get_materials(self, locations):
        material_indices = self.get_region(locations)
        return [self.material_list[i] for i in material_indices]

    def normals(self,direction,locations):
        materials = self.get_materials(locations)
        initial_norm = numpy.zeros((0,3))
        [np.concatenate(initial_norm,mat.normals(direct,locate),axis=0,out=initial_norm)
            for mat, direct, locate in zip(materials, direction, locations)]
        return initial_norm

    def attenuate(self,direction,locations):
        materials = self.get_materials(locations)
        initial_atten = np.zeros((0))
        [np.concatenate(initial_atten,mat.attenuate(direct,locate),axis=0,out=initial_atten)
            for mat, direct, locate in zip(materials, direction, locations)]
        return initial_atten

