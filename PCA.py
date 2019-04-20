import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt

def plotVariance(variance, digit):
    cumulative = np.cumsum(variance)

    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].set_title('Variance feature vectors for digit %d' % digit)
    axs[0].bar(['%s' %i for i in range(len(variance))], variance)
    axs[0].set_xlabel('Individual')
    axs[1].plot(['%s' %i for i in range(len(variance))], cumulative)
    axs[1].set_xlabel('Cummulative')

    plt.draw()

'''
    PCA
    k - number of obtainable features
    mfeat - data points set
    digit - digit for which to obtain the k features

'''


def PCA( mfeat, digit, m):
    #center patterns
    mean = aux.computeMeanVec(mfeat)
    mfeat -= mean

    #compute C
    C = mfeat.dot(mfeat.transpose()) / len(mfeat)

    #compute the SVD decomposition
    U, S, V = np.linalg.svd(C)

    # make sure that the eigenvectors are ordered decreasingly
    pair = np.array([(S[i], U[:, i]) for i in range(len(S))], dtype=object)
    sorted(pair, key=lambda tup: tup[0])
    reversed(pair)
    S = [i[0] for i in pair]
    U = [i[1] for i in pair]

    #compute variance
    sumEigenvalues = np.sum(S)
    if sumEigenvalues != 0:
        variance = [(i/sumEigenvalues)*100 for i in S]
        plotVariance(variance, digit)

    featureVector = []

    U = U[:m]
    mfeat = mfeat.transpose()
    for j in range(m):
        f_j = []
        for i in range(len(U)):
            f_j = np.append(f_j, np.transpose(U[i]).dot(mfeat[j]))
        f_j = np.append(f_j, 1)
        featureVector = np.append(featureVector, f_j)
    (x, y) = mfeat.shape
    featureVector = featureVector.reshape(m, m+1)

    #the decision function is determined by the k feature vectors resulting map
    out = open("Method_4" + "codebook" + str(len(U)) + "number" + str(digit) + ".txt", "w")
    for i in range(m):
        out.write(str(featureVector[i])+"\n")
    out.close()

    return featureVector
