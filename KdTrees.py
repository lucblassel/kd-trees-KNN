"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""
from projet_sort import *
from helpers import *
from plotter import *
from cv import *

import math
import gc

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

@timeit
def timed_create_tree(*args,**kwargs):
    """
    times create_tree function
    """
    return create_tree(*args,**kwargs)

def calculate_dist(point,node):
    """
    returns euclidean distance between 2 points
    """

    if len(point)!=len(node.value):
        return
    vect = np.array(point)-np.array(node.value)
    summed = np.dot(vect,vect)
    return math.sqrt(summed)

def naive_dist(point1,point2):
    vect = np.array(point1)-np.array(point2)
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

@timeit
def timed_nearest_neighbours(*args,**kwargs):
    nearest_neighbours(*args,**kwargs)

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

@timeit
def timed_batch_knn(*args,**kwargs):
    return batch_knn(*args,**kwargs)

def naive_knn(knownPoints,unknownPoints,labelDic,k):
    predictions = []
    for point in unknownPoints:
        distMin = math.inf
        candidates = []
        for known in knownPoints:
            dist = naive_dist(point,known)
            if dist<distMin:
                candidates.append((dist,known))
                candidates.sort(key=lambda point: point[0])
                if len(candidates)>k:
                    candidates.pop()
        candidateslabelsDic = {}
        for node in candidates:
            candidate = tuple(node[1])
            if labelDic[candidate] in candidateslabelsDic:
                candidateslabelsDic[labelDic[candidate]] += 1
            else:
                candidateslabelsDic[labelDic[candidate]] = 1
        predictedLabel = max(candidateslabelsDic, key=candidateslabelsDic.get) #assuming if equality of count each key has a random chance to be the first of this result
        predictions.append(predictedLabel)
    return predictions

@timeit
def timed_naive_knn(*args,**kwargs):
    return naive_knn(*args,**kwargs)

@timeit
def timed_cv(*args,**kwargs):
    return cv(*args,**kwargs)

def main():

    """
    Here we choose what we want to execute
    """
    randomCloud = False
    example = False
    iris = False
    irisCv = False
    leaf = True

    timing = False #if true then functions are timed

    if randomCloud:
        """
        we generate a random dataset with following properties: num points in dims dimensions, with coordinate values contained between min and max.
        we then build a k-d tree of this dataset and print it
        """
        print("\n\n"+100*"="+"\nrandomCloud\n\n")
        num = 10000
        dims = 3
        min = -1000
        max = 1000
        cloud = gen_cloud(num,dims,min,max)
        if timing:
            randomTree = timed_create_tree(cloud,dims)
        else:
            randomTree = create_tree(cloud,dims)

        if num <= 100:
            print(randomTree)

        #calculate nearest neighbours of randomly generated point
        point = gen_cloud(1,dims,min,max)[0]
        candidates = []
        if timing:
            timed_nearest_neighbours(point=point,node=randomTree,candidateList=candidates,k=3)
        else:
            nearest_neighbours(point=point,node=randomTree,candidateList=candidates,k=3)

        print_neighbours(candidates)

    if example:
        """
        we use the data from https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/ to create the trees
        and search for k nearest neighbours for the point to classify
        """
        print("\n\n"+100*"="+"\nexample\n\n")
        dims = 2
        cloud,labels = load_dataset_example()
        labelDic = to_dict(cloud,labels)
        if timing:
            tree = timed_create_tree(cloud,dims)
        else:
            tree = create_tree(cloud,dims)
        print(tree)

        # for just one point
        point = [4,8]
        candidates = []
        if timing:
            timed_nearest_neighbours(point=point,node=tree,candidateList=candidates,k=3)
        else:
            nearest_neighbours(point=point,node=tree,candidateList=candidates,k=3)


        print("nearest neighbours of",point,":")
        print_neighbours(candidates)

        #for multiple points
        cloud2 = [[3, 6],[3, 7],[1, 9]]
        if timing:
            predictions = timed_batch_knn(cloud,cloud2,labelDic,2)
        else:
            predictions = batch_knn(cloud,cloud2,labelDic,2)
            print('naive',naive_knn(cloud,cloud2,labelDic,2))
        print(predictions)

    if iris:
        """
        we test the performance of our method using data from the iris dataset and plots the results
        """
        print("\n\n"+100*"="+"\nIRIS\n\n")
        pointsTrain,targetTrain,pointsTest,targetTest,toPlotTrain,toPlotTest = load_dataset_iris(twoClasses=False)

        pointsDictTrain = to_dict(pointsTrain,targetTrain)
        pointsDictTest = to_dict(pointsTest,targetTest)
        dicIris = {**pointsDictTrain, **pointsDictTest}
        predictions1 = timed_batch_knn(pointsTrain,pointsTest,pointsDictTrain,2)
        predictions2 = timed_naive_knn(pointsTrain,pointsTest,pointsDictTrain,2)
        print_preds(predictions1,pointsDictTest)
        print('naive')
        print_preds(predictions2,pointsDictTest)
        plot_points(toPlotTrain,targetTrain,toPlotTest,predictions)

        if irisCv:
            kList = [1,2,5,10,20]
            cvResultTest,cvResultTrain = cv(pointsTrain,.1,2,kList,dicIris,10)
            print(cvResultTest,cvResultTrain)
            cv_plotter(kList,cvResultTest,cvResultTrain)

    if leaf:
        """
        we test the performance of our algorithm using the leaf dataset and k-fold cross validation, which is in the train.csv file
        we plot the results of the CV. The leaf dataset has a high dimensionality
        """
        print("\n\n"+100*"="+"\nleaf\n\n")
        x,y = load_dataset_leaf()
        dic = to_dict(x,y)

        predictions1 = timed_batch_knn(x[:-20],x[-20:],dic,1)
        print_preds(predictions1,dic)

        kList = [1,2,5,10,20]
        cvResultTest,cvResultTrain=timed_cv(x,.1,10,kList,dic,2)
        cv_plotter(kList,cvResultTest,cvResultTrain)

if __name__=="__main__":
    main()
