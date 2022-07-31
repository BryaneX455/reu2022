# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:36:07 2022

@author: Bryan
"""
import math 
import pandas as pd
import numpy as np
from NNetwork import NNetwork as nn
import networkx as nx
#import utils.NNetwork as nn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning
import warnings
warnings.filterwarnings("ignore")



WSDF = pd.read_csv('GHM_Dict_Data.csv').transpose() 
print(WSDF)
