import numpy as np
from random import seed
import features as feat
import matplotlib.pyplot as plt
import seaborn as sns


"""
RANDOMLY splits the dataset and binary set into n-folds
"""

#100 = 2 * 2 * 5 * 5
# Possible folds {2,4,5,10,20,25,50,100}
def split(dataset, train_y, folds=2):
    t = train_y.tolist()
    zset = []
    splitset = []
    data = list(dataset)
    foldsize = (len(data) / folds)/10
    for i in range(folds):
        fold = []
        zfold = []
        for j in range(0, 10):
            k = 0
            while k < foldsize:
                ind = k + j*len(dataset)//10
                fold.append(data[ind])
                zfold.append(t[ind])
                k += 1
        splitset.append(fold)
        zset.append(zfold)
    return splitset, zset


seed(1)


# print(np.array(kfold(train, 4)).shape)


def ridge(phi, train_y, alpha):
    x = np.transpose(phi).dot(phi)
    id = np.identity(phi[0].size)
    inv = np.linalg.inv(np.add(x, alpha * alpha * id))

    return inv.dot(np.transpose(phi)).dot(train_y)


"""
Calculates the Mean Squared Error
"""


def MSE(d, y):
    return np.sum((d - y) ** 2) / len(d)


def MISS(train_y, train_y_prediction):
    miss = 0
    for i in range(len(train_y)):
        if np.argmax(train_y[i]) != np.argmax(train_y_prediction[i]):
            miss += 1
    return miss/len(train_y)


def linear_regression(train_x, train_y, alpha, test_x, test_y):
    rlist = []  # use this to append the MSE values

    Wopt = ridge(train_x, train_y, alpha)

    train_y_pred = np.array(train_x).dot(Wopt)
    test_y_pred = np.array(test_x).dot(Wopt)

    train_MSE = MSE(train_y_pred, train_y)
    test_MSE = MSE(test_y_pred, test_y)

    train_MISS = MISS(train_y, train_y_pred)
    test_MISS = MISS(test_y, test_y_pred)

    rlist.append(train_MSE)
    rlist.append(test_MSE)
    rlist.append(train_MISS)
    rlist.append(test_MISS)

    return rlist


def kfold(dataset, binaryset, models, alpha):
    """

    :param dataset: this dataset is the set of feature vectors, each with k features
    :param binaryset: this is the set of binary train_y vectors z, where all values
                      are 0 except 1 in index i, for class i
    :param models: this is the model parameters we should modify
                   (feature length, types of features)
    :param alpha: alpha for the ridge regression
    :return:
    """
    # risklist = []
    res = []

    for i in range(len(dataset)):
        set = list(dataset)
        size = len(set[0])
        size *= len(set)

        test_x = np.array(set[i])
        test_y = np.array(binaryset[i])
        train_x = np.array(set)
        train_x = np.delete(train_x, i, 0)
        train_x = np.reshape(train_x, (size-len(test_x), len(set[i][0])))

        train_y = np.array(binaryset)
        train_y = np.delete(np.array(train_y), i, 0)
        train_y = np.reshape(train_y, (size-len(test_x), 10))

        rlist = linear_regression(train_x, train_y, alpha, test_x, test_y)
        res.append(rlist)
    # print("Result is: ", res)
    sum1 = 0
    sum2 = 0
    for i in range(len(res)):
        sum1 = sum1 + res[i][0]
        sum2 = sum2 + res[i][1]

    result = []
    result.append(sum1 / len(res))
    result.append(sum2 / len(res))
    res
    # return np.argmin(res) # for now return the result

    return result


# print(ridge(train, train_y, 0))

#main()

"""
x = np.linalg.pinv(train).dot(train_y)
testvotes = np.array(test).dot(x)
testresults = [max(p) for p in testvotes]
testresults = [x - 1 for x in testresults]

test_y = []
for i in range(10):
    ones = np.zeros((1, 100))
    ones[:] = i
    test_y.append(ones.tolist())

mismatches = np.sum(np.abs(np.sign(np.array(testresults)- np.array(np.ravel(test_y)))))

error = 100*mismatches/1000
"""


def main():
    filename = "mfeat-pix.txt"
    data1 = np.loadtxt(filename)

    w = [100]
    for k in range(1, 100):
        data = feat.createFeatureVectors(k)

        train = data.tolist()
        test = data[1::2]

        train_y = []

        for i in range(10):
            ones = np.zeros((100, 10))
            ones[:, i] = 1
            train_y.append(ones.tolist())
        train_y = np.reshape(np.ravel(train_y), (1000, 10))
        fold = [2, 4, 5, 10, 20, 25, 50]
        foldi = [2]
        for foldSize in fold:

            dataset, binaryset = split(train, train_y, folds=foldSize )
            models = []
            result = kfold(dataset, binaryset, models, 3)
            #print(result)
        #plt.show()

main()
