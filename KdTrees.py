"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""

from projet_sort import *
from random import randint
from datetime import datetime
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

    def hasChildren(self):
        return False if self.right == None and self.left == None else True

    def setVisited(self):
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

def createTree(pointList,dimensions,depth=0,parent=None):
    """
    creates Kd-tree, pointsList is the list of points. dimensions is the dimension of the euclidean space in which these points are present (or number od fimensions along which you want to split the data). depth is the starting tree-depth
    """

    if not pointList:
        return

    if not dimensions:
        dimensions = len(pointList[0]) #selects all dimensions to split along

    axis = depth%dimensions #switch dimensions at each split

    # pointList.sort(key=lambda point: point[axis])
    shellSort(pointList,axis)
    med = len(pointList)//2
    root = Node(value=pointList[med], parent=parent, axis=axis, visited=False)
    root.left = createTree(pointList=pointList[:med],dimensions=dimensions,depth=depth+1, parent=root)
    root.right = createTree(pointList=pointList[med+1:],dimensions=dimensions,depth=depth+1, parent=root)

    return root

def calculateDist(point,node):
    """
    returns euclidean distance between 2 points
    """

    if len(point)!=len(node.value):
        return
    vect = np.array(point)-np.array(node.value)
    summed = np.dot(vect,vect)
    return math.sqrt(summed)

def nearestNeighbours(point,node,candidateList,distMin=math.inf,k=1):

    if node == None:
        return
    elif node.visited:
        return

    dist = calculateDist(point,node)

    if dist < distMin:
        distMin = dist

    #TODO pruning

    candidateList.append([dist,node])
    candidateList.sort(key=lambda point: point[0])

    if len(candidateList)>k:
        print("removing candidates")
        candidateList.pop() #removes last one (biggest distance)

    if  point[node.axis] < node.value[node.axis]:
        nearestNeighbours(point, node.left, candidateList,distMin,k)
        if node.value[node.axis] - point[node.axis] <= distMin:
            nearestNeighbours(point, node.right, candidateList,distMin,k)
        else:
            print("pruned right branch of "+str(node.value))
    else:
        nearestNeighbours(point, node.right, candidateList,distMin,k)
        if point[node.axis] - node.value[node.axis] <= distMin:
            nearestNeighbours(point, node.left, candidateList,distMin,k)
        else:
            print("pruned left branch of "+str(node.value))

    node.visited = True

def printNeighbours(candidates):
    for node in candidates:
        print("node: "+str(node[1].value)+" , distance: "+str(node[0]))

def genCloud(num,dims, min, max):
    return [[randint(min,max) for i in range(dims)] for j in range(num)]

def main():
    num = 100
    dims = 3
    min = -1000
    max = 1000
    # cloud = genCloud(num,dims,min,max)

    #example set from https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/ (FOR TESTING)
    cloud = [[1, 3],[1, 8], [2, 2], [2, 10], [3, 6], [4, 1], [5, 4], [6, 8], [7, 4], [7, 7], [8, 2], [8, 5],[9, 9]]
    dims = 2
    print(datetime.now())
    tree = createTree(pointList=cloud,dimensions=dims)
    print(datetime.now())
    print("tree created")
    print(tree)

    point = [4,8]
    candidates = []
    nearestNeighbours(point=point,node=tree,candidateList=candidates,k=3)
    printNeighbours(candidates)

if __name__=="__main__":
    main()
