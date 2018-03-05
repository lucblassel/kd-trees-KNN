# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-03-05 14:25:57
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-05 14:48:38
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cv_plotter(k_list,accuracy_test,accuracy_train):
	fig, ax = plt.subplots()
	bar_width = 0.35
	opacity = 0.8
	 
	rects1 = plt.bar(k_list, accuracy_train, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 label='Train accuracy')
	 
	rects2 = plt.bar(k_list + np.repeat(bar_width,len(k_list)), accuracy_test, bar_width,
	                 alpha=opacity,
	                 color='g',
	                 label='Test accuracy')
	 
	plt.xlabel('k')
	plt.ylabel('Scores')
	plt.title('CV results')
	plt.xticks(k_list) # + np.repeat(bar_width,len(k_list)), bar_width, tuple(k_list))
	plt.legend()
	plt.tight_layout()
	plt.show()

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
