import numpy as np
import sys
#import reflection

#inputs the input vector and point of origin, and outputs the new direction vector
#normal and direction are numpy arrays of length 3
#point is  numpy array of length 3 as well, except it refers to the point at which the ray had last been reflected
def mirror(vectors,normals):
    reflect= np.subtract(vectors, 2*np.sum(vectors*normals,axis=1)[:,None]*normals)
    return reflect

def points_of_incidence(directions, line_points, normal_vector,surface_point):
    difference=np.subtract(surface_point, line_points)
    dot1=np.sum(difference*normal_vector, axis=1)
    dot2=np.sum(directions*normal_vector, axis=1)
    return  np.add(line_points, (dot1/dot2)*directions)
    
def points_of_incidence2(center, radius, line_points, directions):
    directions=unitize(directions)
    origin_difference=np.subtract(line_points, center)
    a=np.linalg.norm(directions,axis=1)**2
    b=(np.sum(directions*origin_difference, axis=1))
    c=np.linalg.norm(origin_difference, axis=1)**2-radius**2
    distance1=(-b+(b**2-a*c)**0.5)
    distance2=(-b-(b**2-a*c)**0.5)
    return (np.add(line_points,distance1[:,None]*directions),np.add(line_points,distance2[:,None]*directions))

def normal_sphere(points_of_incidence):
    return unitize(points_of_incidence)
    

def unitize(vectors):
    return vectors/np.linalg.norm(vectors,axis=1)[:,None]

def distances(vectors, other_vectors):
    return np.linalg.norm(np.subtract(vectors, other_vectors),axis=1)

def reflect_rays_sphere(center, radius, points, directions):
    incidence_points=points_of_incidence2(center,radius, points, directions)
    boolean_table=np.greater(distances(incidence_points[0],points),distances(incidence_points[1],points))
    incidence_points=np.where(boolean_table[:,None], incidence_points[1],incidence_points[0])            
    normals=normal_sphere(incidence_points)
    new_vectors=unitize(mirror(directions, normals))
    return(incidence_points, new_vectors)
    
def reflect_rays(normals, points, directions, surface_point, surface_normal):
    incidence_points=points_of_incidence(directions,points, surface_normal, surface_point)
    new_vectors=mirror(directions, normals)
    return (incidence_points, new_vectors)

def main():
    number_rays=40
    points=np.tile(np.array([[200,0,0]]),(number_rays,1))
    center=np.array([0,0,0])
    
    directions=np.tile(unitize(np.array([[1,1,0]])),(number_rays,1))
    rands=np.array([np.random.rand(number_rays)/100,np.zeros(number_rays),np.zeros(number_rays)]).T
    
    directions=np.add(directions,rands)
    radius=200
    radius2=220
    
    iterations=40  
    out=np.empty((2*iterations,number_rays,3))
    for i in range(iterations):
        new_stuff=reflect_rays_sphere(center,radius2,points, directions)
        points=new_stuff[0]
        directions=new_stuff[1]

        out[2*i]=points
        
        new_stuff=reflect_rays_sphere(center,radius,points, directions)

        points=new_stuff[0]
        directions=new_stuff[1]
       
        
        out[2*i+1]=points
    
    result=np.concatenate(np.swapaxes(out,0,1))
    print(result)
    np.savetxt('output.csv', result, delimiter=',')
def imain():
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
