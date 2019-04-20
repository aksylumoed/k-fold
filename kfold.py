import numpy as np
from random import seed
from random import randrange

filename="mfeat-pix.txt"

data = np.loadtxt(filename)


train = data[0::2]
test = data[1::2]


def split(dataset, folds=2):
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
            zfold.append(target.pop(ind))
        splitset.append(fold)
        zset.append(zfold)
    return splitset, zset


seed(1)
#print(np.array(kfold(train, 4)).shape)

target = []

for i in range(10):
    ones = np.zeros((100,10))
    ones[:, i] = 1
    target.append(ones.tolist())
target = np.reshape(np.ravel(target), (1000, 10))


def ridge(phi, target, alpha):
    x = np.transpose(phi).dot(phi)
    id = np.identity(len(phi[0]))
    inv = np.linalg.inv(np.add(x, alpha*alpha*id))

    return inv.dot(np.transpose(phi)).dot(target)


def loss(x, y):
    return np.linalg.norm(x-y)


def MSE(d, y):
    return np.sum((d - y)**2)


"""
array m..L would be an list of different model classes
"""
def kfold(dataset, binaryset, models, alpha):
    #risklist = []
    res = []
    for i in range(len(dataset)):
        set = list(dataset)

        validate = set[i]
        corrects = binaryset[i]
        train = set[np.arange(len(set)) != i]
        target = binaryset[np.arange(len(set)) != i]
        rlist = []
        for m in range(1, len(models)+1):
            dopt = ridge(train, target, alpha)

            trainvotes = np.array(train).dot(dopt)
            testvotes = np.array(validate).dot(dopt)
            trainresult = MSE(trainvotes, target)
            testresult = MSE(testvotes, corrects)


            rlist[0].append(trainresult)
            rlist[1].append(testresult)

            #r = (1/len(dataset)) * np.sum(loss(dopt, target))
            #rlist.append(r)

        #risk = np.mean(rlist)
        res.append(rlist)
    print(res)
    return np.argmin(res)


def main():
    dataset, binaryset = split(train, folds=4)

    models = []

    result = kfold(dataset, binaryset, models, 0)

    print(result)


print(ridge(train, target, 0).shape)

"""

x = np.linalg.pinv(train).dot(target)
testvotes = np.array(test).dot(x)
testresults = [max(p) for p in testvotes]
testresults = [x - 1 for x in testresults]

corrects = []
for i in range(10):
    ones = np.zeros((1, 100))
    ones[:] = i
    corrects.append(ones.tolist())

mismatches = np.sum(np.abs(np.sign(np.array(testresults)- np.array(np.ravel(corrects)))))

error = 100*mismatches/1000
"""