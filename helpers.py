"""
author: Luc Blassel
Helper functions for KD-tree implementation of KNN
"""

def printNeighbours(candidates):
    for node in candidates:
        print("node: "+str(node[1].value)+" , distance: "+str(node[0]))

def genCloud(num,dims, min, max):
    return [[randint(min,max) for i in range(dims)] for j in range(num)]

def toDict(points,target):
    """
    converts array of points with array of labels, to dict with label as value and point as key
    """
    if len(points) != len(target):
        raise ValueError("The points and label arrays shoud have the same length.")

    pointsLabelDict = {}
    for i in range(len(points)):
        pointsLabelDict[tuple(points[i])] = target[i]

    return pointsLabelDict

def printPreds(predictions,labelDict):
    c=0
    precision = 0
    for key in labelDict:
        print("predicted: "+str(predictions[c])+"  real: "+str(labelDict[key]))
        if predictions[c] == labelDict[key]:
            precision += 1
        c += 1

    print("precision: "+str(100*precision/c)+"%")
