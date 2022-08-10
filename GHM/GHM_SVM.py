# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:52:36 2022

@author: Zihong Xu
"""

import numpy as np
import pandas   as pd
import seaborn as sb
import networkx as nx
import statistics as st
import matplotlib.pyplot as plt
import random
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


WSDF = pd.read_csv('GHM_Dict_Data_AllStates.csv')
X = np.array(WSDF.loc[:,'kappa':'E_15_15'])
Y = np.array(WSDF['label'])
print(Y)
pass 

ros = RandomOverSampler(random_state=42)
X_resampled, Y_resampled = ros.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size = 0.2, 
                                                    random_state = 4,
                                                    stratify = Y_resampled)


svc = SVC(kernel = 'poly', C = 3.5, random_state = 0)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

conf_matrix = confusion_matrix(y_true = Y_test, y_pred = Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=svc.classes_)
disp.plot()
plt.show()

print('Precision: %.3f' % precision_score(Y_test, Y_pred))
print('Recall: %.3f' % recall_score(Y_test, Y_pred))
print('F1: %.3f' % f1_score(Y_test, Y_pred))
print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred))