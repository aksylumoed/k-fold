import numpy as np
from random import seed
from random import randrange
import features as feat
import matplotlib.pyplot as plt
import seaborn as sns

filename = "mfeat-pix.txt"

data1 = np.loadtxt(filename)

data = feat.createFeatureVectors()
print(data.shape)
train = data.tolist()
test = data[1::2]

train_y = []

for i in range(10):
    ones = np.zeros((100, 10))
    ones[:, i] = 1
    train_y.append(ones.tolist())
train_y = np.reshape(np.ravel(train_y), (1000, 10))


"""
RANDOMLY splits the dataset and binary set into n-folds
"""


def split(dataset, folds=2):
    t = train_y.tolist()
    zset = []
    splitset = []
    data = list(dataset)
    foldsize = int(len(data) / folds)
    for i in range(folds):
        fold = []
        zfold = []
        while len(fold) < foldsize:
            # add random element
            ind = randrange(len(data))
            fold.append(data.pop(ind))
            zfold.append(t.pop(ind))
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

        test_x = np.array(set[i])
        test_y = np.array(binaryset[i])
        train_x = np.array(set)
        train_x = np.delete(train_x, i, 0)
        train_x = np.reshape(train_x, (len(train)-len(test_x), len(set[i][0])))
        print(train_x.shape)
        train_y = np.array(binaryset)
        train_y = np.delete(np.array(train_y), i, 0)
        train_y = np.reshape(train_y, (len(train)-len(test_x), 10))
        print(train_y.shape)

        rlist = []  # use this to append the MSE values
        for m in range(1):
            print("train_x:", train_x)
            print("train_y: ", train_y)
            dopt = ridge(train_x, train_y, alpha)
            print("shape: ", dopt.shape)
            trainvotes = np.array(train_x).dot(dopt)
            testvotes = np.array(test_x).dot(dopt)
            trainresult = MSE(trainvotes, train_y)
            testresult = MSE(testvotes, test_y)

            visualizePredictedAndExpected(test_y, testvotes)

            rlist.append(trainresult)
            rlist.append(testresult)

            # r = (1/len(dataset)) * np.sum(loss(dopt, train_y))
            # rlist.append(r)

        # risk = np.mean(rlist)

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
    # return np.argmin(res) # for now return the result

    return result


def main():
    dataset, binaryset = split(train, folds=4)
    models = []
    result = kfold(dataset, binaryset, models, 3)
    print(result)
    plt.show()


# print(ridge(train, train_y, 0))

main()

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
