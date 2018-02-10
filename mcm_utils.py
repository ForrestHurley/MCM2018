import numpy as np
import math

def normalize(vectors):
    return vectors/np.linalg.norm(vectors,axis=1)[:,np.newaxis]

def dist(vectsA, vectsB):
    return np.linalg.norm(np.subtract(vectsA, vectsB),axis=1)

def dot(vectsA, vectsB):
    return np.sum(vectsA*vectsB,axis=1)

def mirror(rays,normal):
    #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    out = rays - 2*np.expand_dims(dot(rays,normal)/dot(normal,normal),axis=1)*normal
    return out

def random_removal(probability,values):
    indices = np.where(np.random.rand(values[0].shape[0]) < probability)
    return [val[indices] for val in values]

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
                                                             

def newtons_method(func,gradient,guess=(0),iterations=100,error=0.01):
    for i in range(iterations):
        current_error = func(guess)
        if current_error < error:
            return guess
        new_guess = guess - current_error/gradient(guess)
    return guess
