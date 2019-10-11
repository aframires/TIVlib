import numpy as np
from TIVlib import distances

def hchange(tiv_array):
    results = []
    for i in range(len(tiv_array)):
        distance = distances.euclidean(tiv_array[i+1],tiv_array[i])
        results.append(distance)

    return results