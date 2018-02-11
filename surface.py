import mcm_utils
import numpy as np
import materials as mat

class default_surface:
    def __init__(self,material=mat.default_mat):
        self.material = material
    
    def intersection_point(self,ray_origin,ray_direction):
        return ray_origin
    
    def normal_from_surface(self,intersection_location):
        return intersection_location
    
    def reflect_rays(self,ray_origins,ray_directions):
        intersections = self.intersection_point(ray_origins,ray_directions)
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

    def intersection_point(self,ray_origin,ray_direction):
        difference = ray_origin - self.center

        ray_direction = mcm_utils.normalize(ray_direction)

        a = mcm_utils.dot(ray_direction,ray_direction)
        b = mcm_utils.dot(ray_direction,difference)
        c = mcm_utils.dot(difference,difference) - self.radius * self.radius

        discriminant = (b**2-a*c)**0.5

        possible_dist = [-b-discriminant,-b+discriminant]
        abs_dist = np.abs(possible_dist)
        dist = np.where(abs_dist[0] < abs_dist[1],possible_dist[0],possible_dist[1])

        return ray_origin + dist[:,np.newaxis]*ray_direction

    def normal_from_surface(self,intersection_location):
        return intersection_location - self.center

class bumpy_sphere(sphere):
    def __init__(self,material=mat.default_mat,center=[0,0,0],radius=1):
        super().__init__(material,center,radius)

    def normal_intersection(self,ray_origin,ray_direction):
        return super().intersection_point(self,ray_origin,ray_direction)

    def height_of_sphere(self,points):
        return self.radius+sphericals_from_vectors(points)[1]

    def intersection_point(self,ray_origin,ray_direction):
        guess=self.normal_intersection(ray_origin,ray_direction)
        

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
