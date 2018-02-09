import numpy as np
#import reflection

#inputs the input vector and point of origin, and outputs the new direction vector
#normal and direction are numpy arrays of length 3
#point is  numpy array of length 3 as well, except it refers to the point at which the ray had last been reflected
def mirror(vectors,normals):
    return np.subtract(vectors, 2*np.sum(vectors*normals,axis=1)*normals)

def points_of_incidence(directions, line_points, normal_vector,surface_point):
    difference=np.subtract(surface_point, line_points)
    dot1=np.sum(difference*normal_vector, axis=1)
    dot2=np.sum(directions*normal_vector, axis=1)
    return  np.add(line_points, (dot1/dot2)*directions)
    
def points_of_incidence(center, radius, line_points, directions):
    origin_difference=np.subtract(line_points, center)
    term1=-2*(np.sum(directions*origin_difference, axis=1)
    term2=term1**2-4*np.linalg.norm(line_points,ord=1)
def reflect_rays(normals, points, directions, surface_point, surface_normal):
    incidence_points=points_of_incidence(directions,points, surface_normal, surface_point)
    new_vectors=mirror(directions, normals)
    return (incidence_points, new_vectors)

def main():
    surface_point=np.array([0,0,0])
    surface_normal=np.array([0,1,0])
   
    surface_point2=np.array([0,20,0]) 
    surface_normal2=np.array([0,-1,0])

    normals=np.array([surface_normal])
    points=np.array([[0,20,0]])
    directions=np.array([[-1,-1,0]])

    for i in range(10):
        new_stuff=reflect_rays(normals, points, directions, surface_point, surface_normal)
        
        points=new_stuff[0]
        directions=new_stuff[1]
        print(points)
        print(directions)
        
        new_stuff=reflect_rays(normals, points, directions, surface_point2, surface_normal2)
        
        points=new_stuff[0]
        directions=new_stuff[1]
        
        print(points)
        print(directions)
        
        
if __name__=="__main__":
    main()
