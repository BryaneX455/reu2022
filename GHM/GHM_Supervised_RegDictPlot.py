# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 00:49:39 2022

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
from SDL_main.src.SDL_BCD import SDL_BCD
from SDL_main.src.SDL_SVP import SDL_SVP
from NNetwork_master.src.NNetwork import NNetwork as nn
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
from GHM_Classes.Display import Display
from GHM_Classes.NMF import NMF
from itertools import chain
warnings.filterwarnings("ignore")
Display_Class = Display()
NMF_Class = NMF()

WSD = pd.read_csv('GHM_Dict_Data_NWS_AllStates.csv')
"""
NWSEdge Only Prediction using SNMF
"""
print("X", len(WSD.values))

X_Edge = np.array(WSD.loc[:,'E_0_0':'E_24_24'])
Y_Edge = np.array(WSD['label'])
under_sampler = RandomUnderSampler(random_state=42)
X_res_Edge, y_res_Edge = under_sampler.fit_resample(X_Edge, Y_Edge)
X_train_Edge, X_test_Edge, y_train_Edge, y_test_Edge = train_test_split(X_res_Edge, y_res_Edge, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res_Edge)

y_pred_baseline = np.zeros(len(y_test_Edge))    
            
ncol = 5
nrow = 2
fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))          
xi = 1
iter_avg = 1
beta = 0.5
iteration = 100
r = 10
SNMF_Class = SNMF(X=[X_train_Edge.T, y_train_Edge.reshape(-1,1).T],  # data, label
                        X_test=[X_test_Edge.T, y_test_Edge.reshape(-1,1).T],
                        #X_auxiliary = None,
                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                        # ini_code = H_true,
                        xi=xi,  # weight for dim reduction vs. prediction trade-off
                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                        L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                        nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                        full_dim=False)

Result_Dict_Edge = SNMF_Class.train_logistic(iter=iteration, subsample_size=None,
                                                beta = beta,
                                                search_radius_const=np.linalg.norm(X_train_Edge),
                                                update_nuance_param=False,
                                                if_compute_recons_error=False, if_validate=False)

Y_Test_SNMF_Edge = Result_Dict_Edge['Y_test']
Y_Pred_SNMF_Edge = Result_Dict_Edge['Y_pred']
W_Tot = Result_Dict_Edge['loading']
W_Dict_Edge = W_Tot[0]
Reg_Coeff= W_Tot[1]
print(len(Reg_Coeff), len(Reg_Coeff[0]))
H_Edge = Result_Dict_Edge['code']
k = int(np.sqrt(W_Dict_Edge.shape[0]))
dict_shape = (k,k)
axs[0,0].imshow(W_Dict_Edge[:,0].reshape(dict_shape))
axs[0,1].imshow(W_Dict_Edge[:,1].reshape(dict_shape))
axs[0,2].imshow(W_Dict_Edge[:,2].reshape(dict_shape))
axs[0,3].imshow(W_Dict_Edge[:,3].reshape(dict_shape))
axs[0,4].imshow(W_Dict_Edge[:,4].reshape(dict_shape))
axs[1,0].imshow(W_Dict_Edge[:,5].reshape(dict_shape))
axs[1,1].imshow(W_Dict_Edge[:,6].reshape(dict_shape))
axs[1,2].imshow(W_Dict_Edge[:,7].reshape(dict_shape))
axs[1,3].imshow(W_Dict_Edge[:,8].reshape(dict_shape))
axs[1,4].imshow(W_Dict_Edge[:,9].reshape(dict_shape))


axs[0,0].set_title('RC= ' + str(Reg_Coeff[0,1]))
axs[0,1].set_title('RC= ' + str(Reg_Coeff[0,2]))
axs[0,2].set_title('RC= ' + str(Reg_Coeff[0,3]))
axs[0,3].set_title('RC= ' + str(Reg_Coeff[0,4]))
axs[0,4].set_title('RC= ' + str(Reg_Coeff[0,5]))
axs[1,0].set_title('RC= ' + str(Reg_Coeff[0,6]))
axs[1,1].set_title('RC= ' + str(Reg_Coeff[0,7]))
axs[1,2].set_title('RC= ' + str(Reg_Coeff[0,8]))
axs[1,3].set_title('RC= ' + str(Reg_Coeff[0,9]))
axs[1,4].set_title('RC= ' + str(Reg_Coeff[0,10]))
plt.suptitle('NWS')
plt.show()






WSD = pd.read_csv('GHM_Dict_Data_UCLA26_AllStates.csv')

"""
UCLAEdge Only Prediction using SNMF, SDL_BCD Random Forest, and SVM
"""
print("X", len(WSD.values))

X_Edge = np.array(WSD.loc[:,'E_0_0':'E_24_24'])
Y_Edge = np.array(WSD['label'])
under_sampler = RandomUnderSampler(random_state=42)
X_res_Edge, y_res_Edge = under_sampler.fit_resample(X_Edge, Y_Edge)
X_train_Edge, X_test_Edge, y_train_Edge, y_test_Edge = train_test_split(X_res_Edge, y_res_Edge, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res_Edge)

y_pred_baseline = np.zeros(len(y_test_Edge))    
            
# Random Forest  
ncol = 5
nrow = 2
fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))          
xi = 1
iter_avg = 1
beta = 0.5
iteration = 100
r = 10
SNMF_Class = SNMF(X=[X_train_Edge.T, y_train_Edge.reshape(-1,1).T],  # data, label
                        X_test=[X_test_Edge.T, y_test_Edge.reshape(-1,1).T],
                        #X_auxiliary = None,
                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                        # ini_code = H_true,
                        xi=xi,  # weight for dim reduction vs. prediction trade-off
                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                        L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                        nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                        full_dim=False)

Result_Dict_Edge = SNMF_Class.train_logistic(iter=iteration, subsample_size=None,
                                                beta = beta,
                                                search_radius_const=np.linalg.norm(X_train_Edge),
                                                update_nuance_param=False,
                                                if_compute_recons_error=False, if_validate=False)

Y_Test_SNMF_Edge = Result_Dict_Edge['Y_test']
Y_Pred_SNMF_Edge = Result_Dict_Edge['Y_pred']
W_Tot = Result_Dict_Edge['loading']
W_Dict_Edge = W_Tot[0]
Reg_Coeff= W_Tot[1]
print(Reg_Coeff)
H_Edge = Result_Dict_Edge['code']
k = int(np.sqrt(W_Dict_Edge.shape[0]))
dict_shape = (k,k)
axs[0,0].imshow(W_Dict_Edge[:,0].reshape(dict_shape))
axs[0,1].imshow(W_Dict_Edge[:,1].reshape(dict_shape))
axs[0,2].imshow(W_Dict_Edge[:,2].reshape(dict_shape))
axs[0,3].imshow(W_Dict_Edge[:,3].reshape(dict_shape))
axs[0,4].imshow(W_Dict_Edge[:,4].reshape(dict_shape))
axs[1,0].imshow(W_Dict_Edge[:,5].reshape(dict_shape))
axs[1,1].imshow(W_Dict_Edge[:,6].reshape(dict_shape))
axs[1,2].imshow(W_Dict_Edge[:,7].reshape(dict_shape))
axs[1,3].imshow(W_Dict_Edge[:,8].reshape(dict_shape))
axs[1,4].imshow(W_Dict_Edge[:,9].reshape(dict_shape))


axs[0,0].set_title('RC= ' + str(Reg_Coeff[0,1]))
axs[0,1].set_title('RC= ' + str(Reg_Coeff[0,2]))
axs[0,2].set_title('RC= ' + str(Reg_Coeff[0,3]))
axs[0,3].set_title('RC= ' + str(Reg_Coeff[0,4]))
axs[0,4].set_title('RC= ' + str(Reg_Coeff[0,5]))
axs[1,0].set_title('RC= ' + str(Reg_Coeff[0,6]))
axs[1,1].set_title('RC= ' + str(Reg_Coeff[0,7]))
axs[1,2].set_title('RC= ' + str(Reg_Coeff[0,8]))
axs[1,3].set_title('RC= ' + str(Reg_Coeff[0,9]))
axs[1,4].set_title('RC= ' + str(Reg_Coeff[0,10]))
plt.suptitle('UCLA')
plt.show()

WSD = pd.read_csv('GHM_Dict_Data_Caltech36_AllStates.csv')

"""
CaltechEdge Only Prediction using SNMF, SDL_BCD Random Forest, and SVM
"""
print("X", len(WSD.values))

X_Edge = np.array(WSD.loc[:,'E_0_0':'E_24_24'])
Y_Edge = np.array(WSD['label'])
under_sampler = RandomUnderSampler(random_state=42)
X_res_Edge, y_res_Edge = under_sampler.fit_resample(X_Edge, Y_Edge)
X_train_Edge, X_test_Edge, y_train_Edge, y_test_Edge = train_test_split(X_res_Edge, y_res_Edge, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res_Edge)

y_pred_baseline = np.zeros(len(y_test_Edge))    
            
# Random Forest  
ncol = 5
nrow = 2
fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))          
xi = 1
iter_avg = 1
beta = 0.5
iteration = 100
r = 10
SNMF_Class = SNMF(X=[X_train_Edge.T, y_train_Edge.reshape(-1,1).T],  # data, label
                        X_test=[X_test_Edge.T, y_test_Edge.reshape(-1,1).T],
                        #X_auxiliary = None,
                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                        # ini_code = H_true,
                        xi=xi,  # weight for dim reduction vs. prediction trade-off
                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                        L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                        nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                        full_dim=False)

Result_Dict_Edge = SNMF_Class.train_logistic(iter=iteration, subsample_size=None,
                                                beta = beta,
                                                search_radius_const=np.linalg.norm(X_train_Edge),
                                                update_nuance_param=False,
                                                if_compute_recons_error=False, if_validate=False)

Y_Test_SNMF_Edge = Result_Dict_Edge['Y_test']
Y_Pred_SNMF_Edge = Result_Dict_Edge['Y_pred']
W_Tot = Result_Dict_Edge['loading']
W_Dict_Edge = W_Tot[0]
Reg_Coeff= W_Tot[1]
print(Reg_Coeff)
H_Edge = Result_Dict_Edge['code']
k = int(np.sqrt(W_Dict_Edge.shape[0]))
dict_shape = (k,k)
axs[0,0].imshow(W_Dict_Edge[:,0].reshape(dict_shape))
axs[0,1].imshow(W_Dict_Edge[:,1].reshape(dict_shape))
axs[0,2].imshow(W_Dict_Edge[:,2].reshape(dict_shape))
axs[0,3].imshow(W_Dict_Edge[:,3].reshape(dict_shape))
axs[0,4].imshow(W_Dict_Edge[:,4].reshape(dict_shape))
axs[1,0].imshow(W_Dict_Edge[:,5].reshape(dict_shape))
axs[1,1].imshow(W_Dict_Edge[:,6].reshape(dict_shape))
axs[1,2].imshow(W_Dict_Edge[:,7].reshape(dict_shape))
axs[1,3].imshow(W_Dict_Edge[:,8].reshape(dict_shape))
axs[1,4].imshow(W_Dict_Edge[:,9].reshape(dict_shape))


axs[0,0].set_title('RC= ' + str(Reg_Coeff[0,1]))
axs[0,1].set_title('RC= ' + str(Reg_Coeff[0,2]))
axs[0,2].set_title('RC= ' + str(Reg_Coeff[0,3]))
axs[0,3].set_title('RC= ' + str(Reg_Coeff[0,4]))
axs[0,4].set_title('RC= ' + str(Reg_Coeff[0,5]))
axs[1,0].set_title('RC= ' + str(Reg_Coeff[0,6]))
axs[1,1].set_title('RC= ' + str(Reg_Coeff[0,7]))
axs[1,2].set_title('RC= ' + str(Reg_Coeff[0,8]))
axs[1,3].set_title('RC= ' + str(Reg_Coeff[0,9]))
axs[1,4].set_title('RC= ' + str(Reg_Coeff[0,10]))
fig.suptitle('Caltech')
plt.show()

