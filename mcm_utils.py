import numpy

def normalize(vectors):
    return vectors/np.linalg.norm(vectors,axis=1)

def dist(vectsA, vectsB):
    return np.linalg.norm(np.subtract(vectsA, vectsB),axis=1)

def dot(vectsA, vectsB):
    return np.sum(vectsA*vectsB,axis=1)

def mirror(rays,normals):
    #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    return rays - 2*dot(rays,normal)/dot(normal,normal)*normal

def rotation_matrix(a,b):
    #https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    a, b = normalize(a),normalize(b)

    v = np.cross(a,b)
    s = np.linalg.norm(v,axis=1)
    c = dot(a,b)

    I = np.identity(3)

    cross_matrix = [[0,-v[:,2],v[:,1]],
                    [v[:,2],0,-v[:,0]],
                    [-v[:,1],v[:,0],0]]
    cross_matrix = np.transpose(cross_matrix,(2,0,1))

    frac = 1/(1+c)

    R = I + cross_matrix + frac*(cross_matrix @ cross_matrix)
    return R

def rotate_into_frame(frame,vector):
    R = rotation_matrix(frame,[0,0,1])

    rotated = R @ np.expand_dims(vector,axis=2)
    
    return rotated, R
