import mcm_utils
import numpy as np
import materials as mat
import math
import phys_utils
from test import ground

class default_surface:
    def __init__(self,material=mat.default_mat):
        self.material = material
    
    def intersection_point(self,ray_origin,ray_direction):
        return ray_origin
    
    def normal_from_surface(self,intersection_location):
        return intersection_location
    
    def reflect_rays(self,ray_origins,ray_directions):
        intersections = self.intersection_point(ray_origins,ray_directions)
        #print("printing intersections")
        #print(intersections)
        world_normals, material_normals = self.normal(ray_directions,intersections)

        ray_directions, material_normals ,world_normals, intersections = \
            self.material.attenuate(
            ray_directions,material_normals, world_normals,intersections)
        # intersections, world_normals, ray_directions = attenuation(intersections, world_normals, ray_directions)

        new_directions = mcm_utils.mirror(ray_directions,world_normals)

        return intersections, new_directions

    def normal_from_material(self,ray_direction,surface_normal,intersection_location):
        incoming_direction, R = mcm_utils.rotate_into_frame(surface_normal,ray_direction)
        mat_norms = self.material.normals(incoming_direction,intersection_location).T
        R_inv = np.linalg.inv(R)
        return np.squeeze(R_inv @ np.expand_dims(mat_norms,axis=2),axis=2), mat_norms

    def normal(self,ray_direction,intersection_point):
        surface_norm = self.normal_from_surface(intersection_point)
        world_norm, mat_norm = self.normal_from_material(ray_direction,surface_norm,intersection_point)

        return world_norm, mat_norm

    def attenuate_rays(self,ray_direction,material_normal,intersection_location):
        return self.material.attenuate(ray_direction, material_normal, intersection_location)


class sphere(default_surface):
    def __init__(self,material=mat.default_mat,center=[0,0,0],radius=1):
        super().__init__(material)
        self.center = np.array(center)
        self.radius = radius

    def intersection_point(self,ray_origin,ray_direction,radii=None):
        if type(radii)==type(None):
            radii=np.full(ray_origin.shape[0],self.radius)
        
        difference = ray_origin - self.center

        ray_direction = mcm_utils.normalize(ray_direction)

        a = mcm_utils.dot(ray_direction,ray_direction)
        b = mcm_utils.dot(ray_direction,difference)
        c = mcm_utils.dot(difference,difference) - radii * radii

        discriminant = (b**2-a*c)**0.5

        possible_dist = np.array([-b-discriminant,-b+discriminant])
        is_positive=np.greater(possible_dist,np.zeros(possible_dist.shape))
        truth=np.equal(is_positive[0],is_positive[1])
        abs_dist = np.abs(possible_dist)
        case1_dist = np.where(abs_dist[0] < abs_dist[1],possible_dist[0],possible_dist[1])

        case2_dist = possible_dist[1]

        dist=np.where(truth,case1_dist,case2_dist)
                
        return ray_origin + dist[:,np.newaxis]*ray_direction

    def normal_from_surface(self,intersection_location):
        return intersection_location - self.center

class bumpy_sphere(sphere):
    def __init__(self,material=mat.default_mat,center=[0,0,0],radius=1):
        super().__init__(material,center,radius)

    def intersection_point(self,ray_origin,ray_direction):
        earth_radii=np.full(ray_origin.shape[0],6371)
        heights=np.full(ray_origin.shape[0],0)
        error=100
        while abs(error)>0.1:
            radii=earth_radii+heights
            normal_intersection=super().intersection_point(ray_origin,ray_direction,radii)
            new_heights=np.zeros(ray_origin.shape[0])
            latlongs=mcm_utils.geographic_coordinates(normal_intersection).T
            latitudes=latlongs[0]
            longitudes=latlongs[1]
            for k in range(ray_origin.shape[0]):
                new_heights[k]=ground.function_interpolator(latitudes[k],longitudes[k])
            error=np.amax(heights-new_heights)
            heights=new_heights
        return normal_intersection

    def normal_from_surface(self,intersection_location):
        vecs=ground.interpolate_gradient(intersection_location)
        norm = np.append(vecs.T,np.full(vecs.shape[0],-1)).reshape(-1,3)
        print(norm.shape)
        return norm

class ionosphere(sphere):        
    def __init__(self,material=mat.default_mat,center=[0,0,0],radius=6671):
        super().__init__(material,center,radius)
    
    def intersection_point(self,ray_origin,ray_direction):
        earth_radii=np.full(ray_origin.shape[0],6371)
        heights=np.full(ray_origin.shape[0],300)
        error=100
        while error>2:
            print("hi")
            radii=earth_radii+heights
            normal_intersection=super().intersection_point(ray_origin,ray_direction,radii)
            angles=mcm_utils.angle(self.normal_from_surface(normal_intersection),ray_direction)
            sign=np.sign(mcm_utils.dot(self.normal_from_surface(normal_intersection),ray_direction))
            incidence_angle=sign*(np.full(angles.shape,0.5*math.pi)-angles)
            print(mcm_utils.rad2deg(incidence_angle))
            new_heights=np.zeros(angles.shape)
            latlongs=mcm_utils.geographic_coordinates(normal_intersection).T
            latitudes=latlongs[0]
            longitudes=latlongs[1]
            for k in range(angles.shape[0]):
                new_heights[k]=phys_utils.virtual_height(lat=latitudes[k],lon=longitudes[k],theta_i=angles[k])
            error=np.amax(heights-new_heights)
            heights=new_heights

        print("done")
        return normal_intersection

class plane(default_surface):
    def __init__(self,material=mat.default_mat,center=[0,0,0],normal=[0,0,1]):
        super().__init__(material)
        self.center = np.array(center)
        self.normal = np.array(normal)

    def intersection_point(self,ray_origin,ray_direction):
        difference = ray_origin - self.center

        dot1 = mcm_utils.dot(difference,self.normal)
        dot2 = mcm_utils.dot(ray_direction,self.normal)

        return self.center + dot1/dot2*ray_direction

    def normal_from_surface(self,intersection_location):
        return self.normal
