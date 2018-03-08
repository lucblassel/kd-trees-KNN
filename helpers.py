"""
author: Luc Blassel
Helper functions for KD-tree implementation of KNN
"""
import numpy as np
import pandas as pd

import time

from sklearn.datasets import load_iris
from random import randint

def timeit(function):
    """
    decorator that allows us to time a function call
    """
    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)
        te = time.time()
        print ('%r  %2.2f ms' %(function.__name__, (te - ts) * 1000))
        return result

    return timed

def print_neighbours(candidates):
    for node in candidates:
        print("node: "+str(node[1].value)+" , distance: "+str(node[0]))

def gen_cloud(num,dims, min, max):
    return [[randint(min,max) for i in range(dims)] for j in range(num)]

def to_dict(points,target):
    """
    converts array of points with array of labels, to dict with label as value and point as key
    """
    if len(points) != len(target):
        raise ValueError("The points and label arrays shoud have the same length.")

    pointsLabelDict = {}
    for i in range(len(points)):
        pointsLabelDict[tuple(points[i])] = target[i]

    return pointsLabelDict

def print_preds(predictions,labelDict):
    c=0
    precision = 0
    for key in labelDict:
        print("predicted: "+str(predictions[c])+"  real: "+str(labelDict[key]))
        if predictions[c] == labelDict[key]:
            precision += 1
        c += 1

    print("precision: "+str(100*precision/c)+"%")

def load_dataset_iris(twoClasses=True):
    """
    if twoClasses is True we select only classes 0 and 1
    """
    data = load_iris()
    if twoClasses:
        x = data['data']
        y = data['target']
        data['data'] = np.array([x[i] for i in range(len(x)) if y[i] in (2,1)])
        data['target'] = np.array([i for i in y if i in(2,1)])

    randIndex = np.random.choice(len(data['data']),10)

    pointsTrain = np.delete(data['data'],randIndex,0).tolist()
    targetTrain = np.delete(data['target'],randIndex,0).tolist()
    pointsTest = data['data'][randIndex].tolist()
    targetTest = data['target'][randIndex].tolist()

    #selecting columns of iris for plotting
    toPlotTrain = np.delete(data['data'],randIndex,0)[:,[0,2]].tolist()
    toPlotTest = data['data'][randIndex][:,[0,2]].tolist()

    print(pointsTrain,targetTrain)
    print(data['data'][randIndex])

    return pointsTrain,targetTrain,pointsTest,targetTest,toPlotTrain,toPlotTest

def load_dataset_leaf():
    data = pd.read_csv('train.csv',header=0)
    exclude = ['species']
    x = data.loc[:,data.columns.difference(exclude)]
    y = data[['species']]
    return x.as_matrix().tolist(),[i[0] for i in y.values]

def load_dataset_example():
    #example set from https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/
    x = [[1, 3],[1, 8], [2, 2], [2, 10], [3, 6], [4, 1], [5, 4], [6, 8], [7, 4], [7, 7], [8, 2], [8, 5],[9, 9]]
    y = ["Blue", "Blue", "Blue", "Blue", "Blue", "Blue", "Red", "Red", "Red", "Red", "Red", "Red", "Red"]
    return x,y
