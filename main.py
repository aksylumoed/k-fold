import numpy as np
import KmeansInitializations as km
import PCA
import matplotlib.pyplot as plt
import Kfold as kf

filename = "mfeat-pix.txt"
digitImageVectors = np.loadtxt(filename)



# decide which initialization method to use
print("Which initialization method do you want to use?\n")
#method = input("Type 1 for Forgy method, 2 for Random Partition method, or 3 for kcluster ++ \n")
method = 4
F = []
testVectors = []

for digit in range(0, 10):
    left = int(digit)*200
    right = 2 + 200*int(digit)
    mfeat = digitImageVectors[left:right, :5]
    testVectors = np.append(testVectors, mfeat)
    if int(method) in range(1, 4):
        k = input("How many clusters do you want?\n")
        km.kclusters(int(k), mfeat, digit, method)
    else:
        Fd = PCA.PCA(mfeat, digit)
        F.append(Fd)

F = np.reshape(F, (10, 2, len(F[0][0])))
testVectors = np.reshape(testVectors, (20, 5))#len(digitImageVectors[0])))

binarySet = [np.identity(10)[i//2] for i in range(len(testVectors))]
binarySet = np.reshape(binarySet, (len(testVectors), len(binarySet[0])))

kf.Kfold(testVectors, binarySet, 1, F, 2)

plt.show()
