from numpy import *


def main():
    randMat = mat(random.rand(4, 4))
    invRandMat = randMat.I
    print randMat
    print invRandMat
    print randMat*invRandMat

if __name__ == '__main__':
    main()
