import numpy as np
import features as feat
import matplotlib.pyplot as plt

import seaborn as sns


def split(dataset, train_y, folds=2):
    """
    this function will split the dataset into number of *folds*
    :param dataset: set data points (during computation we use feature vectors)
    :param train_y: the binary vectors
    :param folds: number of folds
    :return: a dataset and a binary vector set divided in n folds
    """
    t = train_y.tolist()
    zset = []
    splitset = []
    data = list(dataset)
    foldsize = (len(data) / folds) / 10
    for i in range(folds):
        fold = []
        zfold = []
        for j in range(0, 10):
            k = 0
            while k < foldsize:
                ind = k + j * len(dataset) // 10
                fold.append(data[ind])
                zfold.append(t[ind])
                k += 1
        splitset.append(fold)
        zset.append(zfold)
    return splitset, zset


def ridge(phi, train_y, alpha):
    """

    :param phi: phi is the training data from the feature vector set
    :param train_y: target binary vector set
    :param alpha: we use alpha in order penalise models that overfit
    :return: W_opt that we use to create predicted data points
    """
    x = np.transpose(phi).dot(phi)
    id = np.identity(phi[0].size)
    inv = np.linalg.inv(np.add(x, alpha * alpha * id))

    return inv.dot(np.transpose(phi)).dot(train_y)


def MSE(d, y):
    """
    calculates the Mean Squared Error between two data points
    :param d: vector
    :param y: vector
    :return: mse
    """
    return np.sum(np.power((d - y), 2)) / len(d)


def MISS(train_y, train_y_prediction):
    """
    calculates the missclassification rate between expected and predicted points
    :param train_y:
    :param train_y_prediction:
    :return:
    """
    miss = 0
    for i in range(len(train_y)):
        if np.argmax(train_y[i]) != np.argmax(train_y_prediction[i]):
            miss += 1
    return miss / len(train_y)


def linear_regression(train_x, train_y, alpha, test_x, test_y):
    """
    ridge regression task that computes Wopt using ridge() and calculates
    training and testing error
    :param train_x: training set
    :param train_y: binary target vector for training
    :param alpha:
    :param test_x: validation set
    :param test_y: binary target vector for testing
    :return: list with the training and testing error
    """
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


def kfold(dataset, binaryset, alpha):
    """
    uses an already splitted dataset and picks a new validation and training set
    for each i from 0 to k

    :param dataset: this dataset is the set of feature vectors, each with k features
    :param binaryset: this is the set of binary train_y vectors z, where all values
                      are 0 except 1 in index i, for class i
    :param models: this is the model parameters we should modify
                   (feature length, types of features)
    :param alpha: alpha for the ridge regression
    :return: returns the list of all training and testing error
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
        train_x = np.reshape(train_x, (size - len(test_x), len(set[i][0])))

        train_y = np.array(binaryset)
        train_y = np.delete(np.array(train_y), i, 0)
        train_y = np.reshape(train_y, (size - len(test_x), 10))

        rlist = linear_regression(train_x, train_y, alpha, test_x, test_y)
        res.append(rlist)
    # print("Result is: ", res)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(len(res)):
        sum1 = sum1 + res[i][0]
        sum2 = sum2 + res[i][1]
        sum3 = sum3 + res[i][2]
        sum4 = sum4 + res[i][3]

    result = []
    result.append(sum1 / len(res))
    result.append(sum2 / len(res))
    result.append(sum3 / len(res))
    result.append(sum4 / len(res))
    # return np.argmin(res) # for now return the result

    return result


def main():
    filename = "mfeat-pix.txt"
    data1 = np.loadtxt(filename)
    results = []


    nfeatures = range(1, 241)
    for k in nfeatures:
        print(k)
        data = feat.createFeatureVectors(k)
        train = data.tolist()
        test = data[1::2]

        train_y = []

        for i in range(10):
            ones = np.zeros((100, 10))
            ones[:, i] = 1
            train_y.append(ones.tolist())
        train_y = np.reshape(np.ravel(train_y), (1000, 10))

        # fold = [2, 4, 5, 10, 20, 25, 50]

        # for foldSize in fold:
        # print("For fold: ", foldSize)
        dataset, binaryset = split(train, train_y, folds=5)
        result = kfold(dataset, binaryset, 3)
        # print(result)
        results.append(result)

    print(results)

    mse_train = [results[i][0] for i in range(len(results))]
    mse_test = [results[i][1] for i in range(len(results))]
    miss_train = [results[i][2] for i in range(len(results))]
    miss_test = [results[i][3] for i in range(len(results))]


    """
    
     we use this to plot the training and testing error and to observe effect
      of overfitting and underfitting
  
    """
    sns.set()
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Measuring Training and Testing Error')


    p1, = plt.plot(nfeatures, mse_train, label='MSE_train')
    p2, = plt.plot(nfeatures, mse_test, label='MSE_test')
    p3, = plt.plot(nfeatures, miss_train, label='MISS_train')
    p4, = plt.plot(nfeatures, miss_test, label='MISS_test')
    plt.legend(ncol=2, loc='best', prop={'size': 8})
    plt.show()


main()
