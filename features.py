import numpy as np
from tempfile import TemporaryFile
import help as pca
from sklearn.decomposition import PCA


filename = "mfeat-pix.txt"
digitImageVectors = np.loadtxt(filename)

def bias(transform):
    fin = []
    for i in range(len(transform)):
        fin1 = np.append(transform[i], 1)
        fin = np.append(fin, fin1)
    fin = fin.reshape((len(transform),len(transform[0])+1))
    return fin

def createFeatureVectors(k):
    method = 4
    F = []
    testVectors = []
    for digit in range(0, 10):
        left = int(digit)*200
        right = 100 + 200*int(digit)
        mfeat = digitImageVectors[left:right, :]
        testVectors = np.append(testVectors, mfeat)

        pca = PCA(n_components=k)#(mfeat, digit, k)
        Fd = bias(pca.fit_transform(mfeat))
        print(Fd.shape)
        F.append(Fd)

    # F: Feature vectors
    F = np.reshape(F, (10 * 100 , k+1))
    testVectors = np.reshape(testVectors, (1000, 240))

    return np.reshape(np.ravel(F), (10 * 100, k+1))



# training set


#len(digitImageVectors[0])))

# binary for training set
#binarySet = [np.identity(10)[i//100] for i in range(len(testVectors))]
#binarySet = np.reshape(binarySet, (len(testVectors), len(binarySet[0])))

#print(np.reshape(np.ravel(F), (1000,101)).shape)
