import numpy as np
import math

def normalize(vectors):
    return vectors/np.linalg.norm(vectors,axis=1)[:,np.newaxis]

def dist(vectsA, vectsB,axis=1):
    return np.linalg.norm(np.subtract(vectsA, vectsB),axis=axis)

def dot(vectsA, vectsB):
    return np.sum(vectsA*vectsB,axis=1)

def magnitude(vectsA):
    return np.linalg.norm(vectsA,axis=1)

def angle(vectsA, vectsB):
    ang=np.arccos(np.divide(dot(vectsA, vectsB),magnitude(vectsA)*magnitude(vectsB)))
    return ang

def mirror(rays,normal):
    #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    out = rays - 2*np.expand_dims(dot(rays,normal)/dot(normal,normal),axis=1)*normal
    return out

def local_frame_to_cartesian(from_horizon,from_north,latitude,longitude):
    alpha=(math.pi/180)*(from_north-90)
    beta=(math.pi/180)*(90-from_horizon)
    
    phi=(math.pi/180)*longitude
    theta=(math.pi/180)*(90-latitude)

    theta_hat=np.array([math.cos(theta)*math.cos(phi),math.cos(theta)*math.sin(phi),-math.sin(theta)])
    phi_hat=np.array([-math.sin(phi),math.cos(phi),0])
    r_hat=np.array([math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)])
    return math.cos(alpha)*math.sin(beta)*phi_hat+math.sin(alpha)*math.sin(beta)*theta_hat+math.cos(beta)*r_hat

def random_removal(probability,*values):
    indices = np.where(np.random.rand(values[0].shape[0]) < probability)
    return [val[indices] for val in values]

def lat2theta(latitude):
    return deg2rad(90-latitude)

def long2theta(longitude):
    return deg2rad(longitude)

def sphericals_from_vector(vectors):
    r=np.linalg.norm(vectors,axis=1)
    phi=np.arctan2(vectors.T[1],vectors.T[0])
    theta=np.arccos(np.divide(vectors.T[2],r))
    return np.array([r,phi,theta]).T

def geographic_coordinates(vectors):
    spherical_coordinates=sphericals_from_vector(vectors).T

    latitude=90-rad2deg(spherical_coordinates[2])
    longitude=rad2deg(spherical_coordinates[1])
    return np.array([latitude,longitude]).T  

def cartesian_coordinates(latlongs,radii=None):
    if radii==None:
        radii=np.full(latlongs.shape[0],6371)
    thetas=deg2rad(90-latlongs.T[0])
    phis=deg2rad(latlongs.T[1])
    return cartesians_from_vectors(np.array([radii,phis,thetas]).T)

def cartesians_from_vectors(vectors):
    radii=vectors.T[0]
    x=radii*np.cos(vectors.T[1])*np.sin(vectors.T[2])
    y=radii*np.sin(vectors.T[1])*np.sin(vectors.T[2])
    z=radii*np.cos(vectors.T[2]) 
    return np.array([x,y,z]).T

def add_convolve(vector):
    return np.add(np.delete(vector,0),np.delete(vector,-1))

def rad2deg(radians):
    return radians*(180/math.pi)

def deg2rad(degrees):
    return degrees*(math.pi/180)
def rotation_matrix(a,b):
    #https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    a, b = np.array(a), np.array(b)

    a, b = normalize(a),normalize(b)

    v = np.cross(a,b)
    s = np.linalg.norm(v,axis=1)
    c = dot(a,b)

    I = np.identity(3)

    naught = np.zeros(v.shape[0])

    cross_matrix = np.array([[naught,-v[:,2],v[:,1]],
                            [v[:,2],naught,-v[:,0]],
                            [-v[:,1],v[:,0],naught]])
    cross_matrix = np.transpose(cross_matrix,(2,0,1))

    frac = 1/(1+c)

    R = I + cross_matrix + frac[:,np.newaxis,np.newaxis]*(cross_matrix @ cross_matrix)
    return R

def rotate_into_frame(frame,vector):
    R = rotation_matrix(frame,[[0,0,1]])

    rotated = R @ np.expand_dims(vector,axis=2)
    
    return rotated, R

def orthogonalsFromNormals(normals):#uses nx3 array for normals

    e1 = np.array([np.ones(normals.shape[0]),-normals[:,0]/normals[:,1],np.zeros(normals.shape[0])]).T
    nan_rows = np.where(np.isnan(e1[:,1]))
    inf_rows = np.where(np.isinf(e1[:,1]))

    e1[nan_rows] = [1,0,0]
    e1[inf_rows] = [0,1,0]

    e2 = np.cross(normals,e1)

    e1 = e1/np.expand_dims(np.linalg.norm(e1,axis=1),axis=-1)
    e2 = e2/np.expand_dims(np.linalg.norm(e2,axis=1),axis=-1)

    return e1, e2



