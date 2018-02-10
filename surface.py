from mcm_utils import *
import numpy as np

class surface:
    def __init__(self,material=None,appearance=None):
        self.material = material
    
    def intersection_point(ray_origin,ray_direction):
        return ray_origin
    
    def normal_from_surface(intersection_location):
        return intersection_location
    
    def reflect_rays(ray_origins,ray_directions):
        intersections = intersection_point(ray_origin,ray_direction)
        world_normals, material_normals = normal(ray_directions,intersections)

        attenuation = attenuate_rays(ray_directions,material_normals,intersections)
        intersections, world_normals, ray_directions = attenuation(intersections, world_normals, ray_directions)

        new_directions = mirror(ray_directions,world_normals)

        return intersections, new_directions

    def normal_from_material(ray_direction,surface_normal,intersection_location):
        incoming_direction, R = rotate_into_frame(surface_normal,ray_direction)
        mat_norms = self.material.normals(incoming_direction,intersection_location)
        R_inv = np.linalg.inv(R)
        return R_inv @ np.expand_dims(mat_norms,axis=2), mat_norms

    def normal(ray_direction,intersection_point):
        surface_norm = normal_from_surface(intersection_point)
        world_norm, mat_norm = normal_from_material(ray_direction,surface_norm,intersection_point)

        return world_norm, mat_norm

    def attenuate_rays(ray_direction,material_normal,intersection_location):
        return self.material.attenuate(ray_direction,material_normal,intersection_location)


class sphere(surface):
    def __init(self,material=None,center=[0,0,0],radius=1)
        surface.__init__(material)
        self.center = np.array(center)
        self.radius = radius

    def intersection_point(ray_origin,ray_direction):
        difference = ray_origin - self.center

        a = dot(ray_direction,ray_direction)
        b = dot(ray_direction,difference)
        c = dot(difference,difference) - self.radius * self.radius

        dist = -b + (b**2-a*c)**0.5

        return ray_origin + dist*ray_direction

    def normal_from_surface(intersection_location):
        return intersection_location - center

class plane(surface):
    def __init__(self,material=None,center=[0,0,0],normal=[0,0,1]):
        surface.__init__(material)
        self.center = np.array(center)
        self.normal = np.array(normal)

    def intersection_point(ray_origin,ray_direction):
        difference = ray_origin - self.center

        dot1 = dot(difference,self.normal)
        dot2 = dot(ray_direction,self.normal)

        return self.center + dot1/dot2*ray_direction

    def normal_from_surface(intersection_location):
        return self.normal
