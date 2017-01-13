from numpy import *

# bayes method can be used to classify spam emails, having s set of volcabulary,
# the input of an email ia a 1/0 array indicates wether the words appears in the email
# in order to calculate the conditional probability, we should calculate the prob of every class in trainning
# set together with the conditional prob of word appearance in each class


# trainMatrix in 1,0 matrix of all emails, trainCategory contains classification info pf each email
def tarinNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pSpam = sum(trainCategory)/float(numTrainDocs)
    # avoiding zero values in array
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # avoid numbers too small
    p1Vect = math.log(p1Num/p1Denom)
    p0Vect = math.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pSpam


def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + math.log(pClass1)
    p0 = sum(vec2Classify * p1Vect) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

