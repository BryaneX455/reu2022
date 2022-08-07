# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:13:28 2022

@author: Bryan
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from NNetwork import NNetwork as nn
from sklearn import svm
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
from sklearn.datasets import make_sparse_coded_signal
from SDL_src.SNMF import SNMF
warnings.filterwarnings("ignore")

WSD = pd.read_csv('GHM_Dict_Data.csv')