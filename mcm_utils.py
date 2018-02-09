import numpy

def normalize(vectors):
    return vectors/np.linalg.norm(vectors,axis=1)

def dist(vectsA, vectsB):
    return np.linalg.norm(np.subtract(vectsA, vectsB),axis=1)

def dot(vectsA, vectsB):
    return np.sum(vectsA*vectsB,axis=1)
