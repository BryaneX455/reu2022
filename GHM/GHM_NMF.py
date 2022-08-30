# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:36:07 2022

@author: Bryan
"""

#Imports
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from tqdm import trange
from GHM_Classes.Display import Display
from GHM_Classes.NMF import NMF
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore")

NMF_Class = NMF()
Display_Class = Display()



    
"""
NWS Graph GHM
"""
WSD = pd.read_csv('GHM_Dict_Data_NWS_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title1 = 'Adj_Matrix for NWS_GHM Sync'
title2 = 'Adj_Matrix for NWS_GHM Non-Sync'

# Edge only



c_True = df_True.loc[:3972, 'E_0_0':'E_24_24']
c_False = df_False.loc[:,'E_0_0':'E_24_24']
X_True = c_True.values.transpose()
X_False = c_False.values.transpose()
W_True, H_True = NMF_Class.ALS(X=X_True, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_True: {X_True.shape}\nShape of W_True: {W_True.shape}\nShape of H_True: {H_True.shape}")

W_False, H_False = NMF_Class.ALS(X=X_False, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)



print(f"Shape of X_False: {X_False.shape}\nShape of W_False: {W_False.shape}\nShape of H_False: {H_False.shape}")
Display_Class.display_dictionary(title = title1, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title = title2, W=X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-25-walks-NWS-importance',
                       W=W_True, 
                       At = np.dot(H_True, H_True.T), 
                       fig_size=[20,10], 
                       show_importance=True)

Display_Class.display_dict_and_graph(title='W_False-25-walks-NWS-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)





        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'NWS_GHM_Dict_to_Graph_Deg')

true_norm = np.linalg.norm(W_True.T, ord=1, axis=1)
false_norm = np.linalg.norm(W_False.T, ord=1, axis=1)

data_arr = np.stack((true_norm, false_norm), axis=0).T
df_boxplot = pd.DataFrame(data_arr, columns = ['Synchronizing','Non-Synchronizing'])

ax = sb.boxplot(data=df_boxplot, saturation=1)
ax.axes.set_title("L$_1$ norm for NWS NMF Dictionaries", fontsize=15)
plt.ylabel("L$_1$ norm", labelpad=10)
plt.show()
true_norm_real = np.linalg.norm(c_True.T, ord=1, axis=1)
false_norm_real = np.linalg.norm(c_False.T, ord=1, axis=1)

data_arr_real = np.stack((true_norm_real, false_norm_real), axis=0).T
df_boxplot_real = pd.DataFrame(data_arr_real, columns = ['Synchronizing','Non-Synchronizing'])

ax = sb.boxplot(data=df_boxplot_real, saturation=1)
ax.axes.set_title("L$_1$ norm for NWS NMF Dictionaries", fontsize=15)
plt.ylabel("L$_1$ norm", labelpad=10);
plt.show()
"""
UCLA Graph GHM
"""

WSD = pd.read_csv('GHM_Dict_Data_UCLA26_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title3 = 'Adj_Matrix for UCLA26_GHM Sync'
title4 = 'Adj_Matrix for UCLA26_GHM Non-Sync'
# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:623, 'E_0_0':'E_24_24']
c_False = df_False.loc[:,'E_0_0':'E_24_24']
X_True = c_True.values.transpose()
X_False = c_False.values.transpose()
W_True, H_True = NMF_Class.ALS(X=X_True, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_True: {X_True.shape}\nShape of W_True: {W_True.shape}\nShape of H_True: {H_True.shape}")


W_False, H_False = NMF_Class.ALS(X=X_False, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_False: {X_False.shape}\nShape of W_False: {W_False.shape}\nShape of H_False: {H_False.shape}")
Display_Class.display_dictionary(title3, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title4, X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-25-walks-UCLA-importance',
                       W=W_True, 
                       At = np.dot(H_True, H_True.T), 
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-25-walks-UCLA-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)



        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'UCLA26_GHM_Dict_to_Graph_Deg')

true_norm = np.linalg.norm(W_True.T, ord=1, axis=1)
false_norm = np.linalg.norm(W_False.T, ord=1, axis=1)

data_arr = np.stack((true_norm, false_norm), axis=0).T
df_boxplot = pd.DataFrame(data_arr, columns = ['Synchronizing','Non-Synchronizing'])
df_boxplot

ax = sb.boxplot(data=df_boxplot, saturation=1)
ax.axes.set_title("L$_1$ norm for UCLA NMF Dictionaries", fontsize=15)
plt.ylabel("L$_1$ norm", labelpad=10)


"""
Caltech Graph GHM
"""
WSD = pd.read_csv('GHM_Dict_Data_Caltech36_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title3 = 'Adj_Matrix for Wisconsin87_GHM Sync'
title4 = 'Adj_Matrix for Wisconsin87_GHM Non-Sync'
# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:, 'E_0_0':'E_24_24']
c_False = df_False.loc[:4778,'E_0_0':'E_24_24']
X_True = c_True.values.transpose()
X_False = c_False.values.transpose()
W_True, H_True = NMF_Class.ALS(X=X_True, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)



W_False, H_False = NMF_Class.ALS(X=X_False, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_False: {X_False.shape}\nShape of W_False: {W_False.shape}\nShape of H_False: {H_False.shape}")
Display_Class.display_dictionary(title3, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title4, X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-25-walks-Caltech-importance',
                       W=W_True, 
                       At = np.dot(H_True, H_True.T), 
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-25-walks-Caltech-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)




        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'Caltech36_GHM_Dict_to_Graph_Deg')

true_norm = np.linalg.norm(W_True.T, ord=1, axis=1)
false_norm = np.linalg.norm(W_False.T, ord=1, axis=1)

data_arr = np.stack((true_norm, false_norm), axis=0).T
df_boxplot = pd.DataFrame(data_arr, columns = ['Synchronizing','Non-Synchronizing'])
df_boxplot

ax = sb.boxplot(data=df_boxplot, saturation=1)
ax.axes.set_title("L$_1$ norm for Caltech NMF Dictionaries", fontsize=15)
plt.ylabel("L$_1$ norm", labelpad=10);


