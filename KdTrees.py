"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""

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

def createTree(pointList,dimensions):

    if not pointList:
        raise ValueError("Points list must be provided...")

    
