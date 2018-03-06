"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""
from projet_sort import *
from helpers import *
from plotter import *
from cv import *

import math

import numpy as np

class Node:
    """
    Kd-tree Node
    """

    def __init__(self,value=None,parent=None,left=None,right=None,axis=None,visited=False):
        self.value = value #coordinates of point
        self.parent = parent
        self.left = left
        self.right = right
        self.axis = axis
        self.visited = visited

    def has_children(self):
        return False if self.right == None and self.left == None else True

    def set_visited(self):
        self.visited=True

    def __str__(self, depth=0):
        """
        modified snippet of Steve Krenzel
        """
        dim = len(self.value)
        ret = ""

        # Print right branch
        if self.right != None:
            ret += self.right.__str__(depth + 1)

        # Print own value
        ret += "\n" + ("    "*depth*dim) + str(self.value)

        # Print left branch
        if self.left != None:
            ret += self.left.__str__(depth + 1)

        return ret

    def reset(self):
        """
        sets all visited values to false
        """
        self.visited = False

        if self.right:
            self.right.reset()
        if self.left:
            self.left.reset()

def create_tree(pointList,dimensions,depth=0,parent=None):
    """
    creates Kd-tree, pointsList is the list of points. dimensions is the dimension of the euclidean space in which these points are present (or number od fimensions along which you want to split the data). depth is the starting tree-depth
    """

    if not pointList:
        return

    if not dimensions:
        dimensions = len(pointList[0]) #selects all dimensions to split along

    axis = depth%dimensions #switch dimensions at each split

    # pointList.sort(key=lambda point: point[axis])
    # shellSort(pointList,axis)
    quicksort(pointList,0,len(pointList)-1,axis)

    med = len(pointList)//2
    root = Node(value=pointList[med], parent=parent, axis=axis, visited=False)
    root.left = create_tree(pointList=pointList[:med],dimensions=dimensions,depth=depth+1, parent=root)
    root.right = create_tree(pointList=pointList[med+1:],dimensions=dimensions,depth=depth+1, parent=root)

    return root

def calculate_dist(point,node):
    """
    returns euclidean distance between 2 points
    """

    if len(point)!=len(node.value):
        return
    vect = np.array(point)-np.array(node.value)
    summed = np.dot(vect,vect)
    return math.sqrt(summed)

def nearest_neighbours(point,node,candidateList,distMin=math.inf,k=1,verbose=False):

    if node == None:
        return
    elif node.visited:
        return

    dist = calculate_dist(point,node)

    if dist < distMin:
        candidateList.append([dist,node])
        candidateList.sort(key=lambda point: point[0])
        distMin = candidateList[-1][0]


    if len(candidateList)>k:
        if verbose:
            print("removing candidates")
        candidateList.pop() #removes last one (biggest distance)

    if  point[node.axis] < node.value[node.axis]:
        nearest_neighbours(point, node.left, candidateList,distMin,k)
        if node.value[node.axis] - point[node.axis] <= distMin:
            nearest_neighbours(point, node.right, candidateList,distMin,k)
        else:
            if verbose:
                print("pruned right branch of "+str(node.value))
    else:
        nearest_neighbours(point, node.right, candidateList,distMin,k)
        if point[node.axis] - node.value[node.axis] <= distMin:
            nearest_neighbours(point, node.left, candidateList,distMin,k)
        else:
            if verbose:
                print("pruned left branch of "+str(node.value))

    node.visited = True

def batch_knn(knownPoints,unknownPoints,labelDic,k):
    tree = create_tree(pointList=knownPoints,dimensions=len(knownPoints[0]))
    predictions = []
    for point in unknownPoints:
        # print(point)
        candidates =[]
        nearest_neighbours(point=point,node=tree,candidateList=candidates,k=k)
        candidateslabelsDic = {}
        for node in candidates:
            candidate = tuple(node[1].value)
            if labelDic[candidate] in candidateslabelsDic:
                candidateslabelsDic[labelDic[candidate]] += 1
            else:
                candidateslabelsDic[labelDic[candidate]] = 1
        predictedLabel = max(candidateslabelsDic, key=candidateslabelsDic.get) #assuming if equality of count each key has a random chance to be the first of this result
        predictions.append(predictedLabel)
        tree.reset()
    return predictions

def main():
    num = 100
    dims = 3
    min = -1000
    max = 1000
    cloud = gen_cloud(num,dims,min,max)

    #testing with iris dataset
    # pointsTrain,targetTrain,pointsTest,targetTest,toPlotTrain,toPlotTest = load_dataset_iris()
    #
    # pointsDictTrain = to_dict(pointsTrain,targetTrain)
    # pointsDictTest = to_dict(pointsTest,targetTest)
    # dicIris = {**pointsDictTrain, **pointsDictTest}

    x,y = load_dataset_leaf()
    dic = to_dict(x,y)

    kList = [1,2,5,10,20]
    cvResultTest,cvResultTrain=cv(x,.1,10,kList,dic,2)

    cloud,labels = load_dataset_example()
    cloud2 = [[3, 6],[3, 7],[1, 9]]
    point = [4,8]
    labelDic = to_dict(cloud,labels)
    dims = 2

    # tree = create_tree(cloud,dims)
    # candidates = []
    # nearest_neighbours(point,tree,candidates,k=3)
    # nearest_neighbours(point=point,node=tree,candidateList=candidates,k=3)
    # print_neighbours(candidates)
    # predictions = batch_knn(pointsTrain,pointsTest,pointsDictTrain,1)
    # print("Predicted classes : ",predictions)
    # plot_points(toPlotTrain,targetTrain,toPlotTest,predictions)
    #predictions = batch_knn(pointsTrain,pointsTest,pointsDictTrain,2)
    #print_preds(predictions,pointsDictTest)
    # cvResultTest,cvResultTrain = cv(pointsTrain,.1,2,kList,dicIris,10)
    cv_plotter(kList,cvResultTest,cvResultTrain)

if __name__=="__main__":
    main()
