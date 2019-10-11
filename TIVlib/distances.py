import numpy as np

def euclidean(tiv1,tiv2):
    return np.linalg.norm(tiv1.vector-tiv2.vector)

def cosine(tiv1,tiv2):
    tiv1_split = np.concatenate((tiv1.vector.real, tiv1.vector.iamg), axis=0)
    tiv2_split = np.concatenate((tiv2.vector.real,tiv2.vector.iamg),axis=0)
    return np.arccos(np.dot(tiv1_split,tiv2_split)/(np.linalg.norm(tiv1.vector)*np.linalg.norm(tiv2.vector)))