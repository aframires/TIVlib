import numpy as np

def mags(tiv):
    return np.abs(tiv.vector)

def diatonicity(tiv):
    return mags(tiv)[4] / tiv.weights[4]

def wholetoneness(tiv):
    return mags(tiv)[5] / tiv.weights[5]

def chromaticity(tiv):
    return mags(tiv)[0] / tiv.weights[0]