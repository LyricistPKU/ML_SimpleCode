from numpy import *


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


# use all data in each cycle
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weights = ones((n, 1))
    for k in range(maxCycle):
        h = sigmid(dataMatrix*weights)
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
    return weights


# use only one data to update weights
def stoGradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n, 1))
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = labelMat[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights


# modified stoGradAscent
def stoGradAscent1(dataMatIn, classLabels, numIter = 150):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    m, n = shape(dataMatrix)
    weights = ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha decrease when i,j  increase, easier to converse
            alpha = 4/(1.0+i+j)+0.01
            # use a random example
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = (sum(dataMatrix[randIndex]*weights))
            error = labelMat[randIndex] - h
            weights += alpha * error * dataMatrix[i]
            del dataIndex[randIndex]
    return weights

