import numpy as np
from random import seed
from random import randrange
import features as feat


filename="mfeat-pix.txt"

data1 = np.loadtxt(filename)

data = feat.createFeatureVectors()
print(data.shape)
train = data.tolist()
test = data[1::2]

target = []

for i in range(10):
    ones = np.zeros((100,10))
    ones[:, i] = 1
    target.append(ones.tolist())
target = np.reshape(np.ravel(target), (1000, 10))


def split(dataset, folds=2):
    t = target.tolist()
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
#print(np.array(kfold(train, 4)).shape)




def ridge(phi, target, alpha):
    x = np.transpose(phi).dot(phi)
    id = np.identity(phi[0].size)
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

        validate = np.array(set[i])
        corrects = np.array(binaryset[i])
        train = np.array(set)
        train = np.delete(train, i, 0)
        train = np.reshape(train, (750, 101))
        print(train.shape)
        target = np.array(binaryset)
        target = np.delete(np.array(target), i, 0)
        target = np.reshape(target, (750, 10))
        print(target.shape)
        rlist = []
        for m in range(1):

            print("Train:", train)
            print("Target: ", target)
            dopt = ridge(train, target, alpha)

            trainvotes = np.array(train).dot(dopt)
            testvotes = np.array(validate).dot(dopt)
            trainresult = MSE(trainvotes, target)
            testresult = MSE(testvotes, corrects)


            rlist.append(trainresult)
            rlist.append(testresult)

            #r = (1/len(dataset)) * np.sum(loss(dopt, target))
            #rlist.append(r)

        #risk = np.mean(rlist)

        res.append(rlist)
    #print("Result is: ", res)
    sum1 = 0
    sum2= 0
    for i in range(len(res)):
        sum1 = sum1 + res[i][0]
        sum2 = sum2 + res[i][1]

    result = []
    result.append(sum1/len(res))
    result.append(sum2/len(res))
    #return np.argmin(res) # for now return the result

    return result


def main():
    dataset, binaryset = split(train, folds=4)

    models = []

    result = kfold(dataset, binaryset, models, 0)

    print(result)


#print(ridge(train, target, 0))

main()


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

