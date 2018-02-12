import numpy as np
from customocean import waves
import scipy
import mcm_utils
from phys_utils import *

class default_mat:
    def __init__(self,albedo=1):
        self.albedo = albedo

    def attenuate(self,directions, mat_norms, world_norms, intersections):
        return mcm_utils.random_removal(self.albedo,directions, mat_norms, world_norms, intersections )

    def normals(self,direction,location):
        return np.array([*np.zeros((2,location.shape[0])),np.ones(location.shape[0])])

    def set_albedo(self, albedo):
        self.albedo = albedo

class fresnelMaterial(default_mat):
    def __init__(self,index_of_refraction=1.33):
        super().__init__()
        self.ref_index = index_of_refraction

    def attenuate(self, directions, mat_norms, world_norms, intersections):
       
        dir_mags = np.sqrt(mcm_utils.dot(directions, directions))

        normal_mags = np.sqrt(mcm_utils.dot(mat_norms, mat_norms))

        thetas = np.arccos(mcm_utils.dot(directions, mat_norms)/(dir_mags*normal_mags))

        indices = np.where(thetas > np.pi / 2)
        thetas[indices] = np.pi - thetas[indices]

        super().set_albedo(index_reflectance_array(thetas, self.ref_index))
        b = super().attenuate(directions, mat_norms, world_norms, intersections)
        return(b)
    def get_ind(self):
        return self.ref_index

    def attenuate_ind(self, index):
        self.ref_index = index


class simpleWater(default_mat):
    def __init__(self,turbulence=0.1, salinity=35, temp=17):
        super().__init__()
        self.water_model = waves()
        self.salinity = salinity
        self.temp = temp

    def normals(self,direction,location):
        return self.water_model.getRandomNormals(direction).T

    def attenuate(self, *args):
        return super().attenuate(*args)


class simpleAtmosphere(default_mat):
    pass


class fresnelWater(fresnelMaterial):
    def __init__(self, index_of_refraction=7.5, salinity=35, temp=17, omega=1e7, water_model = waves()):
        super().__init__(index_of_refraction)
        self.ref_index = index_of_refraction
        self.salinity = salinity
        self.temp = temp
        self.omega = omega

        self.set_empirical()
        self.water_model = water_model

    def normals(self,direction,location):
        return self.water_model.getRandomNormals(direction).T

    def set_plasma(self):
        self.ref_index = water_plasma_index(self.salinity, self.omega)

    def set_empirical(self):
        self.ref_index = water_empirical_index(self.salinity, self.temp, self.omega)

    def attenuate(self, *args):
        super().attenuate_ind(self.ref_index)
        return super().attenuate(*args)


class fresnelAtmosphere(simpleAtmosphere):
    def __init__(self, index_of_refraction=0.5):
        super().__init__(index_of_refraction)


class physicalWater(simpleWater):
    pass

class physicalAtmosphere(default_mat):
    def __init__(self, frequency=1e7, year=2000, month=12, hour=12, day=1): 
        super().__init__()
        self.frequency = frequency
        self.year = year
        self.month = month
        self.hour = hour
        self.day = day

    
    def is_day_ray(self, ray):
        return is_day(ray[0], ray[1], self.day, self.month, self.year, self.hour)


    def attenuate(self, *args):
        #unchanged_indices = np.where(not self.is_day_ray(args[3]))

        directions = args[0] 
        dir_mags = np.sqrt(mcm_utils.dot(directions, directions))

        normals = args[1]
        normal_mags = np.sqrt(mcm_utils.dot(normals, normals))

        thetas = np.arccos(mcm_utils.dot(directions, normals)/(dir_mags*normal_mags))

        db_loss = D_layer_loss(freq=self.frequency, slice_sizes=10, thetas=thetas)
        #db_loss[unchanged_indices] = 0
        super().set_albedo(10**(-1*db_loss))
        return super().attenuate(*args)



    

class fresnelDirt(fresnelMaterial):
    def __init__(self, index_of_refraction=.5):
        super().__init__(index_of_refraction)
        self.ref_index = index_of_refraction

    def recalculate_indices(self, rays, omega=1e6):
        indices = []
        coords = mcm_utils.geographic_coordinates(rays)
        for coord in coords:
            indices.append(earth_surface_index(coord[0], coord[1], omega))
        
        self.ref_index = np.array(indices)

    def attenuate(self, *args):
        self.recalculate_indices(args[3])
        super().attenuate_ind(self.ref_index)
        return super().attenuate(*args)


class landWaterMat(default_mat):
    def __init__(self, water_model, land_model, year=2000, month=12, day=1, hour=12):
        super().__init__()
        self.water_model = water_model
        self.land_model = land_model

        self.year = year
        self.month = month
        self.hour = hour
        self.day = day


    def choose_model(self, ray):
        return is_ground_array(ray[0, :], ray[1, :])

    def attenuate(self, *args):
        ground_indices = np.where(self.choose_model(args[3]))
        water_indices = np.where(not self.choose_model(args[3]))

        ground_waves = [val[ground_indices] for val in args]
        water_waves = [val[water_indices] for val in args]
        
        out1 = self.land_model.attenuate(ground_waves)
        out2 = self.water_model.attenuate(water_waves)

        return np.concatenate(out1, out2)


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

