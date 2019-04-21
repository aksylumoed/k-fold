import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
from sklearn.decomposition import PCA


data = pd.read_csv("mfeat-pix.txt",sep='\s+', header=None)

darr = data.values.astype(float)

def img_cat(darr):
    """
    reshape the image and show the image
    """
    img_mat = darr.reshape(16, 15) # reshape the d array
    plt.imshow(img_mat, cmap='gray')
    plt.show()
def imgs_cat(darr):
    for rows in darr:
        img_cat(rows)



def add_bais(X):
    # get the dimension
    N, D = X.shape
    Y = np.ones((N, D + 1))
    Y[:,:-1] = X
    return Y
def square_norm(x):
    return np.sum(np.power(x, 2))

def onehot_encode(digit):
    rst = np.zeros(10)
    rst[digit] = 1
    return rst

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
            print("j"+str(j))
            print(foldsize)
            print(len(dataset))
            while k < foldsize:
                ind = k + j * len(dataset) // 10
                print(k)
                print(ind)
                fold.append(data[ind])
                zfold.append(t[ind])
                k += 1
        splitset.append(fold)
        zset.append(zfold)
    return splitset, zset

def linear_regression(X, Xtest, Y, Ytest, alpha=0):

    # calculate the optimal weight
    Wopt = np.matmul(np.matmul(la.inv(np.matmul(X.T, X)), X.T), Y).T

    # calculate the training error term
    # first make the prediction
    Ypred = np.matmul(Wopt, X.T).T
    Ytestpred = np.matmul(Wopt, Xtest.T).T

    # calculate the error
    mse_train = square_norm(Ypred - Y) / 1000.0
    num_miss_train = 0
    for i in range(1000):
        if np.argmax(Ypred[i]) != np.argmax(Y[i]):
            num_miss_train = num_miss_train + 1
    miss_train = num_miss_train / 1000.0

    mse_test = square_norm(Ytestpred - Ytest) / 1000.0
    num_miss_test = 0
    for i in range(1000):
        if np.argmax(Ytestpred[i]) != np.argmax(Ytest[i]):
            num_miss_test = num_miss_test + 1
    miss_test = num_miss_test / 1000.0

    rlist = []
    rlist.append(mse_train)
    rlist.append( mse_test)
    rlist.append(miss_train)
    rlist.append(miss_test)
    return rlist

def kp(dataset, binaryset, alpha=0):
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

        rlist = linear_regression(train_x, train_y, test_x, test_y)
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
    # do a linear regression on feature of number k
    filename = "mfeat-pix.txt"
    data = np.loadtxt(filename)

    results = []

    nfeatures = range(1, 101)
    for k in nfeatures:
        pca = PCA(n_components=k)
        data_pca = pca.fit_transform(data)

        train = add_bais(data_pca)
        train_y = []

        for i in range(10):
            ones = np.zeros((100, 10))
            ones[:, i] = 1
            train_y.append(ones.tolist())
        train_y = np.reshape(np.ravel(train_y), (1000, 10))

        # split the training set
        dataset, binaryset = split(train, train_y, folds=5)
        result = kp(dataset, binaryset)
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







