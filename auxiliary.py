import math
import numpy as np

'''
    computeMean
    kcluster -  array containing the cluster corresponding to the data point 
                at the same position in the initial array
                i.e kcluster[i] =  cluster of mfeat[i] 
    clusterSize - the size of each cluster
                i.e. clusterSize[i] = | cluster i |
    mfeat - array containing the data points

    The function computes the mean of all vectors in a cluster 
    which represent the respective codebook vector
    i.e. the sum of all vectors in the cluster, divided by the 
    number of elements in that cluster 
'''


def computeMean(kcluster, clusterSize, mfeat):
    cb_vectors = np.zeros(len(clusterSize) * len(mfeat[0])).reshape(len(clusterSize), len(mfeat[0]))
    for i in range(len(mfeat)):
        cb_vectors[int(kcluster[i])] += mfeat[i]
    for i in range(len(clusterSize)):
        if clusterSize[i] == 0:
            cb_vectors[i] = np.zeros(len(mfeat[0]))
        else:
            cb_vectors[i] /= clusterSize[i]
    return cb_vectors

'''
    compute mean - version 2
    mfeat - array containing the datapoints
    Function computes the mean of those datapoints
'''


def computeMeanVec(mfeat):
    mean = np.zeros(len(mfeat[0]))
    for i in range(len(mfeat)):
        mean += mfeat[i]
    return mean/len(mfeat)


'''
    euclideanDistance
    dataPoint - the current dataPoint
    codebookVector -  the current codebook vector

    The function computes the euclidean distance between the two elements
    i.e the radical of the sum of all the powers of 2 of the subtractions 
    between elements at the same position in vector 
'''


def euclideanDistance(dataPoint, codebookVector):
    d = 0.0
    for i in range(len(dataPoint)):
        d += (dataPoint[i] - codebookVector[i]) ** 2
    return math.sqrt(d)


'''
    kpp
    set - array of data points
    k -  number of cluster
    function computes the initial codebook vectors
'''


def kpp(set, k):
    # take random center, we'll take the first point
    c = [set[0]]
    for k in range(1, k):
        '''
        for all the points x in the dataset, find the distance
        between the x and the nearest center already chosen
        '''
        d2 = get_distance(set, c)
        '''
        find new data point based off of the weighted prob. 
        distribution where a point x is chosen with prob. 
        proportional to d^2
        '''
        probs = d2/d2.sum()
        '''
        we find the cumulative sums, so that when we get a random
        int r, we can pick the index i that such that 
        cumsums[i-1] < r < cumsums[i]
        '''
        cumsums = probs.cumsum()
        # check if we need to bound the rand int by the total prob
        r = np.random.rand()
        for j,p in enumerate(cumsums):
            if r < p:
                i = j
                break
        # OR
        '''
        i = np.random.choice(len(probs), 1, p=probs)
        c.append(set[i[0]])
        '''
        c.append(set[i])
    return c

'''
    get_distance
    S - set of all data points
    C - set of all codeBook Vectors
    This function returns the set of the minimum distances from 
    each point to a cluster
'''


def get_distance(S, C):
    result = []
    for s in S:
        products = []
        for c in C:
            products.append(np.inner(c-s, c-s))
        result.append(min(products))

    return np.array(result)
