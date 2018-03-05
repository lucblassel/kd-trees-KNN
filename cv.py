# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-03-05 14:46:35
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-05 14:50:56
import numpy as np
from KdTrees import *

def cv_splitter(original_set_len,k):
    splits = []
    if k>original_set_len:
        fold_size = 1
    else :
        fold_size = original_set_len//k
    # print("original_set_len : ",original_set_len)
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
    # print ("all_indexes : ",all_indexes)
    # print ("test_indexes : ",test_indexes)
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
    cv_result_test = []
    cv_result_train = []
    for k_nn in range_k_nn:
        for rep in range(reps):
            d = 0
            print("cv rep number : ",rep+1)
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
        cv_result_train.append(np.mean(acc_results_cv))
        predictions_test = batch_knn(train_set,test_set,label_dic,k_nn)
        cv_result_test.append(accuracy(test_set_labels,predictions_test))
        print("ending cv at mean inner test accuracy : ",cv_result_train[-1]," test acc : ",cv_result_test[-1])
    return cv_result_test,cv_result_train

def accuracy(y_true,y_pred):
    bool_res = []
    for i in range(len(y_true)):
        bool_res.append(y_true[i] == y_pred[i])
    int_res = list(map(int,bool_res))
    accuracy = np.sum(int_res)/len(y_true)
    return accuracy
