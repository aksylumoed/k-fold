import numpy as np
from tempfile import TemporaryFile
import PCA as pca


filename = "mfeat-pix.txt"
digitImageVectors = np.loadtxt(filename)

def createFeatureVectors():
    method = 4
    F = []
    testVectors = []
    k = 40
    for digit in range(0, 10):
        left = int(digit)*200
        right = 100 + 200*int(digit)
        mfeat = digitImageVectors[left:right, :]
        testVectors = np.append(testVectors, mfeat)

        Fd = pca.PCA(mfeat, digit, k)
        F.append(Fd)

    # F: Feature vectors
    F = np.reshape(F, (10, k, k+1))
    testVectors = np.reshape(testVectors, (1000, 240))

    return np.reshape(np.ravel(F), (10 * k, k+1))



# training set


#len(digitImageVectors[0])))

# binary for training set
#binarySet = [np.identity(10)[i//100] for i in range(len(testVectors))]
#binarySet = np.reshape(binarySet, (len(testVectors), len(binarySet[0])))

#print(np.reshape(np.ravel(F), (1000,101)).shape)
