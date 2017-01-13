from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistanceIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistanceIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

你喜欢
# normalize the dataSet
def autoNorm(dataSet):
    # find the min and max value in columns
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = minVals - maxVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = linalg.solve(normDataSet, tile(ranges, (m, 1)))
    return normDataSet, ranges, minVals


def main():
    dataSet, labels = createDataSet()
    inX = [0, 0]
    print classify(inX, dataSet, labels, 3)

if __name__ == '__main__':
    main()
