# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-03-05 14:25:57
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-05 14:48:38

import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

def cv_plotter(kList,accuracyTest,accuracyTrain):
	fig, ax = plt.subplots()
	barWidth = 0.35
	opacity = 0.8

	rects1 = plt.bar(kList, accuracyTrain, barWidth,
	                 alpha=opacity,
	                 color='b',
	                 label='Train accuracy')

	rects2 = plt.bar(kList + np.repeat(barWidth,len(kList)), accuracyTest, barWidth,
	                 alpha=opacity,
	                 color='g',
	                 label='Test accuracy')

	plt.xlabel('k')
	plt.ylabel('Scores')
	plt.title('CV results')
	plt.xticks(kList) # + np.repeat(barWidth,len(kList)), barWidth, tuple(kList))
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_points(knownPoints,knownLabels,unknownPoints,predictedLabels):
    xKnown,yKnown = zip(*knownPoints)
    xUnknown,yUnknown = zip(*unknownPoints)
    #df_known = pd.DataFrame({'x' : xKnown, 'y' : yKnown, 'color' : knownLabels})
    #df_unknown = pd.DataFrame({'x' : xUnknown, 'y' : yUnknown, 'color' : predictedLabel})
    colorLabels = list(set(knownLabels))
    rgbValues = sns.color_palette("Set2", 100)
    colorMap = dict(zip(colorLabels, rgbValues))
    colorsKnown = []
    colorsUnknown = []
    pprint(knownLabels)
    pprint(colorLabels)
    pprint(colorMap)

    for knownLabel in knownLabels:
        colorsKnown.append(colorMap[knownLabel])
    for predictedLabel in predictedLabels:
        colorsUnknown.append(colorMap[predictedLabel])

    plt.scatter(xKnown, yKnown, c=colorsKnown)
    plt.scatter(xUnknown, yUnknown, c=colorsUnknown,marker='+')
    plt.show()
