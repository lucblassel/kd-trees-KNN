"""
KNN implementation using KD-trees
authors: Luc Blassel, Romain Gautron
"""

from projet_sort import *
from helpers import *
from random import randint
from datetime import datetime
from sklearn.datasets import load_iris

import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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

    def reset(self):
        """
        sets all visited values to false
        """
        self.visited = False

        if self.right:
            self.right.reset()
        if self.left:
            self.left.reset()

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

def batch_knn(known_points,unknown_points,label_dic,k):
    tree = createTree(pointList=known_points,dimensions=len(known_points[0]))
    predictions = []
    for point in unknown_points:
        print(point)
        candidates =[]
        nearestNeighbours(point=point,node=tree,candidateList=candidates,k=k)
        candidates_labels_dic = {}
        for node in candidates:
            candidate = tuple(node[1].value)
            if label_dic[candidate] in candidates_labels_dic:
                candidates_labels_dic[label_dic[candidate]] += 1
            else:
                candidates_labels_dic[label_dic[candidate]] = 1
        predicted_label = max(candidates_labels_dic, key=candidates_labels_dic.get) #assuming if equality of count each key has a random chance to be the first of this result
        predictions.append(predicted_label)
        tree.reset()
    return predictions

def plot_points(known_points,known_labels,unknown_points,predicted_labels):
    x_known,y_known = zip(*known_points)
    x_unknown,y_unknown = zip(*unknown_points)
    #df_known = pd.DataFrame({'x' : x_known, 'y' : y_known, 'color' : known_labels})
    #df_unknown = pd.DataFrame({'x' : x_unknown, 'y' : y_unknown, 'color' : predicted_label})
    color_labels = list(set(known_labels))
    rgb_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, rgb_values))
    colors_known = []
    colors_unknown = []

    for known_label in known_labels:
        colors_known.append(color_map[known_label])
    for predicted_label in predicted_labels:
        colors_unknown.append(color_map[predicted_label])

    plt.scatter(x_known, y_known, c=colors_known)
    plt.scatter(x_unknown, y_unknown, c=colors_unknown,marker='+')
    plt.show()

def cv_splitter(original_set_len,k):
    splits = []
    if k>original_set_len:
        fold_size = 1
    else :
        fold_size = original_set_len//k
    print("original_set_len : ",original_set_len)
    indexes =  list(range(original_set_len))
    while len(indexes) >= fold_size:
        split = np.random.choice(indexes, size=fold_size, replace=False).tolist()
        splits.append(split)
        indexes = list(set(indexes)-set(split))
    if len(indexes) > 0:
        splits[-1].extend(indexes)
    return splits

def train_test_splitter(original_set_len,test_percentage):
    test_indexes = []
    all_indexes =  list(range(original_set_len))
    N_test = int(np.round(original_set_len*test_percentage))
    test_indexes = np.random.choice(all_indexes, size=N_test, replace=False).tolist()
    return test_indexes

def test_to_train_indexes(original_set_len,test_indexes):
    all_indexes = list(range(original_set_len))
    print ("all_indexes : ",all_indexes)
    print ("test_indexes : ",test_indexes)
    train_indexes = list(set(all_indexes)-set(test_indexes))
    return train_indexes

def cv(known_points,test_percentage,k_fold,range_k_nn,label_dic,reps):
    acc_results_cv=[]
    original_set_len = len(known_points)
    test_indexes = train_test_splitter(original_set_len,test_percentage)
    train_indexes = test_to_train_indexes(original_set_len,test_indexes)
    test_set = [known_points[i] for i in test_indexes]
    test_set_labels = []
    for test_point in test_set:
        test_set_labels.append(label_dic[tuple(test_point)])
    train_set = [known_points[i] for i in train_indexes]
    train_set_len = len(train_set)
    c = 0
    cv_result_test = []
    cv_result_train = []
    for k_nn in range_k_nn:
        for rep in range(reps):
            d = 0
            print("cv rep number : ",c+1)
            test_cv_indexes_list = cv_splitter(train_set_len,k_fold)
            for test_cv_indexes in test_cv_indexes_list:
                print("cv fold number : ",d+1)
                train_cv_indexes = test_to_train_indexes(train_set_len,test_cv_indexes)
                test_cv_set = [train_set[i] for i in test_cv_indexes]
                train_cv_set = [train_set[i] for i in train_cv_indexes]
                test_cv_labels = []
                for test_cv_point in test_cv_set:
                    test_cv_labels.append(label_dic[tuple(test_cv_point)])
                predictions_cv = batch_knn(train_cv_set,test_cv_set,label_dic,k_nn)
                acc_cv = accuracy(test_cv_labels,predictions_cv)
                acc_results_cv.append(acc_cv)
                d += 1
            c += 1
        cv_result_train.append(np.mean(acc_results_cv))
        predictions_test = batch_knn(train_cv_set,test_cv_set,label_dic,k_nn)
        cv_result_test.append(accuracy(test_set_labels,predictions_test))
        print("ending cv at mean inner test accuracy : ",mean_acc_cv," test acc : ",acc_test)
    return cv_result_test,cv_result_train

def accuracy(y_true,y_pred):
    bool_res = []
    for i in range(len(y_true)):
        bool_res.append(y_true[i] == y_pred[i])
    int_res = list(map(int,bool_res))
    accuracy = np.sum(int_res)/len(y_true)
    return accuracy

def main():
    num = 100
    dims = 3
    min = -1000
    max = 1000
    # cloud = genCloud(num,dims,min,max)

    #testing with iris dataset
    data = load_iris()
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

    pointsDictTrain = toDict(pointsTrain,targetTrain)
    pointsDictTest = toDict(pointsTest,targetTest)
    dicIris = {**pointsDictTrain, **pointsDictTest}

    #example set from https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/ (FOR TESTING)
    cloud = [[1, 3],[1, 8], [2, 2], [2, 10], [3, 6], [4, 1], [5, 4], [6, 8], [7, 4], [7, 7], [8, 2], [8, 5],[9, 9]]
    cloud2 = [[3, 6],[3, 7],[1, 9]]
    labels = ['A', 'A', 'B', 'B', 'C', 'A', 'C', 'B', 'B', 'C', 'A', 'A', 'A']
    label_dic = {(1, 3):"A",(1, 8):"A", (2, 2):"B", (2, 10):"B", (3, 6):"C", (4, 1):"A", (5, 4):"C", (6, 8):"B", (7, 4):"B", (7, 7):"C", (8, 2):"A", (8, 5):"A",(9, 9):"A"}
    dims = 2

    # candidates = []
    # nearestNeighbours(point=point,node=tree,candidateList=candidates,k=3)
    # printNeighbours(candidates)
    predictions = batch_knn(pointsTrain,pointsTest,pointsDictTrain,1)
    print("Predicted classes : ",predictions)
    plot_points(toPlotTrain,targetTrain,toPlotTest,predictions)
    #predictions = batch_knn(pointsTrain,pointsTest,pointsDictTrain,2)
    #printPreds(predictions,pointsDictTest)
    print(cv(pointsTrain,.1,2,2,dicIris,10))

if __name__=="__main__":
    main()
