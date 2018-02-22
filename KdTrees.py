"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""

from projet_sort import *
from random import randint
import sys

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

def createTree(pointList,dimensions,depth=0,parent=None):
    """
    creates Kd-tree, pointsList is the list of points. dimensions is the dimension of the euclidean space in which these points are present (or number od fimensions along which you want to split the data). depth is the starting tree-depth
    """
    print(pointList)

    if not pointList:
        return

    if not dimensions:
        dimensions = len(pointList[0]) #selects all dimensions to split along

    axis = depth%dimensions #switch dimensions at each split

    pointList = shellSort(pointList,axis)
    med = len(pointList)//2

    root = Node(value=pointList[med], parent=parent, axis=axis, visited=False)
    root.left = createTree(pointList=pointList[:med],dimensions=dimensions,depth=depth+1, parent=root)
    root.right = createTree(pointList=pointList[med+1:],dimensions=dimensions,depth=depth+1, parent=root)

    return root

def genCloud(num,dims, min, max):
    return [[randint(min,max) for i in range(dims)] for j in range(num)]

def main():
    num = 10
    dims = 2
    min = -5
    max = 5
    cloud = genCloud(num,dims,min,max)

    tree = createTree(pointList=cloud,dimensions=dims)
    print(tree)

if __name__=="__main__":
    main()
