# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:33:04 2022

@author: Bryan
"""

import random
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import seaborn as sb
from SDL_main.src.SNMF import SNMF
from NNetwork import NNetwork as nn
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
from sklearn.datasets import make_sparse_coded_signal
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC
from karateclub import Node2Vec
from karateclub import Graph2Vec

def generate_data(df):
    df_true = df[df['label'] == True]
    df_false = df[df['label'] == False]
    
    if len(df_true) >= len(df_false):
        df_true = df_true.sample(n = len(df_false))
    else:
        df_false = df_false.sample(n = len(df_true))
        
    df = pd.concat([df_true, df_false], ignore_index = True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.25)
    
    return X_train, X_test, y_train, y_test

WSD = pd.read_csv('GHM_Dict_Data_AllStates.csv')
X = WSD.loc[:,'E_0_0':'E_15_15']
Y = WSD['label']
df = pd.concat([X, Y], axis=1)
df.reset_index(drop=True)
X_train, X_test, y_train, y_test = generate_data(df)
clf = rf(max_depth=5, max_features="log2")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
conf_matrix_RF_G2V = confusion_matrix(y_true = y_test, y_pred = y_pred)
tn = conf_matrix_RF_G2V[0, 0]
tp = conf_matrix_RF_G2V[1, 1]
fn = conf_matrix_RF_G2V[1, 0]
fp = conf_matrix_RF_G2V[0, 1]
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_RF_G2V)
print((tn+tp)/(tn+tp+fn+fp))
disp.plot()
plt.show()
