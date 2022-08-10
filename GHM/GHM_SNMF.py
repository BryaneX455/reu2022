# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:13:28 2022

@author: Bryan
"""
#Imports
import random
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from SDL_main.src.SNMF import SNMF
from NNetwork import NNetwork as nn
from sklearn import svm
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
# from yellowbrick.datasets import load_occupancy
# from yellowbrick.model_selection import FeatureImportances
# from karateclub import Graph2Vec
warnings.filterwarnings("ignore")

def twoRandomNumbers(a,b): 
    test = random.random() # random float 0.0 <= x < 1.0 
    if test < 0.5: return a 
    else: return b

def is_square(i: int) -> bool:
    return i == math.isqrt(i) ** 2

def coding(X, W, H0, 
          r=None, 
          a1=0, #L1 regularizer
          a2=0, #L2 regularizer
          sub_iter=[5], 
          stopping_grad_ratio=0.0001, 
          nonnegativity=True,
          subsample_ratio=1):
    """
    Find \hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) within radius r from H0
    Use row-wise projected gradient descent
    """
    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[1])
    if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
        idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
    A = W.T @ W ## Needed for gradient computation
    grad = W.T @ (W @ H0 - X)
    while (i < np.random.choice(sub_iter)):
        step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
        H1 -= step_size * grad 
        if nonnegativity:
            H1 = np.maximum(H1, 0)  # nonnegativity constraint
        i = i + 1
    return H1

def ALS(X,
        n_components = 10, # number of columns in the dictionary matrix W
        n_iter=100,
        a0 = 0, # L1 regularizer for H
        a1 = 0, # L1 regularizer for W
        a12 = 0, # L2 regularizer for W
        H_nonnegativity=True,
        W_nonnegativity=True,
        compute_recons_error=False,
        subsample_ratio = 10):
    
        '''
        Given data matrix X, use alternating least squares to find factors W,H so that 
                                || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
        is minimized (at least locally)
        '''
        
        d, n = X.shape
        r = n_components
        
        #normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
        #print('!!! avg entry of X', normalization)
        #X = X/normalization

        # Initialize factors 
        W = np.random.rand(d,r)
        H = np.random.rand(r,n) 
        # H = H * np.linalg.norm(X) / np.linalg.norm(H)
        for i in trange(n_iter):
            #H = coding_within_radius(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            #W = coding_within_radius(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            H = coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            W = coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            W /= np.linalg.norm(W)
            if compute_recons_error and (i % 10 == 0) :
                print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
        return W, H

def display_dictionary(W, save_name=None, score=None, grid_shape=None, figsize=[10,10]):
    k = int(np.sqrt(W.shape[0]))
    rows = int(np.sqrt(W.shape[1]))
    cols = int(np.sqrt(W.shape[1]))
    if grid_shape is not None:
        rows = grid_shape[0]
        cols = grid_shape[1]
    
    figsize0=figsize
    if (score is None) and (grid_shape is not None):
        figsize0=(cols, rows)
    if (score is not None) and (grid_shape is not None):
        figsize0=(cols, rows+0.2)
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize0,
                            subplot_kw={'xticks': [], 'yticks': []})
        
    for ax, i in zip(axs.flat, range(W.shape[1])):
        if score is not None:
            idx = np.argsort(score)
            idx = np.flip(idx)    
            
            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="viridis", interpolation='nearest')
            ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)
        else: 
            ax.imshow(W.T[i].reshape(k, k), cmap="viridis", interpolation='nearest')
            if score is not None:
                ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)
       
    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    if save_name is not None:
        plt.savefig( save_name, bbox_inches='tight')
    plt.show()


WSD = pd.read_csv('GHM_Dict_Data_AllStates.csv')
# print(len(WSD.columns))
# Later_States_Start = WSD.columns.get_loc("S_15_29")
# print("Later States Start:", Later_States_Start)
Num_Iter_Pick = 20
Accuracy_SNMF = list(np.zeros(Num_Iter_Pick))
Accuracy_RF = list(np.zeros(Num_Iter_Pick))
Accuracy_SVM = list(np.zeros(Num_Iter_Pick))
Accuracy_Baseline = list(np.zeros(Num_Iter_Pick))
print(len(WSD.values[0,:]))
feature_names = [f"feature {i}" for i in range(len(WSD.values[0,:]))]
for i in range(Num_Iter_Pick):
    X = np.array(WSD.iloc[:,1:(278+i*16)])
    Y = np.array(WSD['label'])
    under_sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = under_sampler.fit_resample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                        test_size = 0.2, 
                                                        random_state = 4, 
                                                        stratify = y_res)
    
    y_pred_baseline = np.zeros(len(y_test))
    # Baseline Predicator
    for j in range(len(y_test)):
        if i == 0:
            if np.all((X_test[j,7:23] == 0)):
                y_pred_baseline[j] = 1
            else:
                y_pred_baseline[j] = twoRandomNumbers(0,1)
        else:
            if np.all((X_test[j,278+(i-1)*16:278+i*16] == 0)):
                y_pred_baseline[j] = 1
            else:
                y_pred_baseline[j] = twoRandomNumbers(0,1)
    conf_matrix_baseline = confusion_matrix(y_true = y_test, y_pred = y_pred_baseline)           
    tn_baseline = conf_matrix_baseline[0, 0]
    tp_baseline = conf_matrix_baseline[1, 1]
    fn_baseline = conf_matrix_baseline[1, 0]
    fp_baseline = conf_matrix_baseline[0, 1]
    accuracy_baseline = (tp_baseline + tn_baseline) / (tp_baseline + tn_baseline + fp_baseline + fn_baseline)           
    Accuracy_Baseline[i] = accuracy_baseline         
                
    # Random Forest            
    xi = 1
    iter_avg = 1
    beta = 0.5
    iteration = 100
    r = 10
    SNMF_Class = SNMF(X=[X_train.T, y_train.reshape(-1,1).T],  # data, label
                            X_test=[X_test.T, y_test.reshape(-1,1).T],
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
    
    Result_Dict = SNMF_Class.train_logistic(iter=iteration, subsample_size=None,
                                                    beta = beta,
                                                    search_radius_const=np.linalg.norm(X_train),
                                                    update_nuance_param=False,
                                                    if_compute_recons_error=False, if_validate=False)
    
    # print(Result_Dict)
    print(i)
    Y_Test_SNMF = Result_Dict['Y_test']
    Y_Pred_SNMF = Result_Dict['Y_pred']
    W_Dict = Result_Dict['loading']
    Sample_Row_Num = math.isqrt(len(W_Dict[0])) ** 2
    Sample_Reg_Num = math.isqrt(len(W_Dict[1])) ** 2
    print("Dict:", W_Dict[0])
    display_dictionary(W_Dict[0][np.random.choice(len(W_Dict[0]), Sample_Row_Num, replace=False), :], figsize=[10,10])
    # display_dictionary(W_Dict[1][np.random.choice(len(W_Dict[1]), Sample_Reg_Num, replace=False), :], figsize=[10,10])
    print("Reg_Coeff:", W_Dict[1])
    Accuracy_SNMF[i] = Result_Dict['Accuracy']
    # conf_matrix_RF = confusion_matrix(y_true = Y_Test_SNMF[0,:], y_pred = Y_Pred_SNMF)
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  # display_labels = None)
    

    clf = rf(max_depth=5, max_features="log2")
    clf.fit(X_train, y_train)
    y_pred_RF = clf.predict(X_test)
    conf_matrix_RF = confusion_matrix(y_true = y_test, y_pred = y_pred_RF)
    tn = conf_matrix_RF[0, 0]
    tp = conf_matrix_RF[1, 1]
    fn = conf_matrix_RF[1, 0]
    fp = conf_matrix_RF[0, 1]
    accuracy_RF = (tp + tn) / (tp + tn + fp + fn)
    Accuracy_RF[i] = accuracy_RF
    important_features_dict = {}
    importances = clf.feature_importances_
    for idx, val in enumerate(importances):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)

    print(f'10 most important features: {important_features_list[:10]}')
    for k in range(10):
        print(WSD.columns[important_features_list[k]])
        
    All_Features_List = list(np.zeros(len(X_train[0])))
    for l in range(len(X_train[0])):
        All_Features_List[l] = WSD.columns[l]   
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0).reshape((-1,1))
    feature_names = list(All_Features_List)

    forest_importances = pd.Series(importances, index=feature_names)
    if len(forest_importances) > 100:
        forest_importances = forest_importances.sort_values(ascending=False).head(50)

    fig, ax = plt.subplots(figsize=(10,8))
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    """
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    """




    #SVM
    svc = SVC(kernel = 'poly', C = 3.5, random_state = 0)
    svc.fit(X_train, y_train)
    y_pred_SVM = svc.predict(X_test)
    conf_matrix_SVM = confusion_matrix(y_true = y_test, y_pred = y_pred_SVM)

    tn_SVM = conf_matrix_SVM[0, 0]
    tp_SVM = conf_matrix_SVM[1, 1]
    fn_SVM = conf_matrix_SVM[1, 0]
    fp_SVM = conf_matrix_SVM[0, 1]
    accuracy_SVM = (tp_SVM + tn_SVM) / (tp_SVM + tn_SVM + fp_SVM + fn_SVM)
    Accuracy_SVM[i] = accuracy_SVM


# disp.plot()
plt.show()
plt.plot(Accuracy_SNMF, label = "SNMF")
plt.plot(Accuracy_RF, label = "RF")
plt.plot(Accuracy_SVM, label = "SVM")
plt.plot(Accuracy_Baseline, '--',label = "baseline")
plt.legend(loc="lower right")
plt.show()

# Dictionary graph， regression coeff
# spectral decomposition
# graph2vec
