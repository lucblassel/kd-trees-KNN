# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-03-05 14:46:35
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-05 14:50:56
import numpy as np
from KdTrees import *

def cv_splitter(originalSetLen,k):
    splits = []
    if k>originalSetLen:
        foldSize = 1
    else :
        foldSize = originalSetLen//k
    # print("originalSetLen : ",originalSetLen)
    indexes =  list(range(originalSetLen))
    while len(indexes) >= foldSize:
        split = np.random.choice(indexes, size=foldSize, replace=False).tolist()
        splits.append(split)
        indexes = list(set(indexes)-set(split))
    if len(indexes) > 0:
        splits[-1].extend(indexes)
    return splits

def train_test_splitter(originalSetLen,testPercentage):
    testIndexes = []
    allIndexes =  list(range(originalSetLen))
    nTest = int(np.round(originalSetLen*testPercentage))
    testIndexes = np.random.choice(allIndexes, size=nTest, replace=False).tolist()
    return testIndexes

def test_to_train_indexes(originalSetLen,testIndexes):
    allIndexes = list(range(originalSetLen))
    # print ("allIndexes : ",allIndexes)
    # print ("testIndexes : ",testIndexes)
    trainIndexes = list(set(allIndexes)-set(testIndexes))
    return trainIndexes

def cv(knownPoints,testPercentage,kFold,rangeKNn,labelDic,reps,naive=False):
    accResultsCv=[]
    originalSetLen = len(knownPoints)
    testIndexes = train_test_splitter(originalSetLen,testPercentage)
    trainIndexes = test_to_train_indexes(originalSetLen,testIndexes)
    testSet = [knownPoints[i] for i in testIndexes]
    testSetLabels = []
    for testPoint in testSet:
        testSetLabels.append(labelDic[tuple(testPoint)])
    trainSet = [knownPoints[i] for i in trainIndexes]
    trainSetLen = len(trainSet)
    cvResultTest = []
    cvResultTrain = []
    for kNn in rangeKNn:
        for rep in range(reps):
            d = 0
            print("cv rep number : ",rep+1)
            testCvIndexesList = cv_splitter(trainSetLen,kFold)
            for testCvIndexes in testCvIndexesList:
                print("cv fold number : ",d+1)
                trainCvIndexes = test_to_train_indexes(trainSetLen,testCvIndexes)
                testCvSet = [trainSet[i] for i in testCvIndexes]
                trainCvSet = [trainSet[i] for i in trainCvIndexes]
                testCvLabels = []
                for testCvPoint in testCvSet:
                    testCvLabels.append(labelDic[tuple(testCvPoint)])
                    if not naive:
                        predictionsCv = batch_knn(trainCvSet,testCvSet,labelDic,kNn)
                    else:
                        predictionsCv = naive_knn(trainCvSet,testCvSet,labelDic,kNn)
                accCv = accuracy(testCvLabels,predictionsCv)
                accResultsCv.append(accCv)
                d += 1
        cvResultTrain.append(np.mean(accResultsCv))
        predictions_test = batch_knn(trainSet,testSet,labelDic,kNn)
        cvResultTest.append(accuracy(testSetLabels,predictions_test))
        print("ending cv at mean inner test accuracy : ",cvResultTrain[-1]," test acc : ",cvResultTest[-1])
    return cvResultTest,cvResultTrain

def accuracy(yTrue,yPred):
    boolRes = []
    for i in range(len(yTrue)):
        boolRes.append(yTrue[i] == yPred[i])
    intRes = list(map(int,boolRes))
    accuracy = np.sum(intRes)/len(yTrue)
    return accuracy
