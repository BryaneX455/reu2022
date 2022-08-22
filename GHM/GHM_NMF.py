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
import warnings
from tqdm import trange
from GHM_Classes.Display import Display
from GHM_Classes.NMF import NMF
warnings.filterwarnings("ignore")

NMF_Class = NMF()
Display_Class = Display()



    
"""
WS Graph GHM
"""
WSD = pd.read_csv('GHM_Dict_Data_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title1 = 'Adj_Matrix for WS_GHM Sync'
title2 = 'Adj_Matrix for WS_GHM Non-Sync'

# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:, 'E_0_0':'E_15_15']
c_False = df_False.loc[:,'E_0_0':'E_15_15']
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
Display_Class.display_dict_and_graph(title='W_True-20-walks-WS-importance',
                       W=W_True, 
                       At = np.dot(H_True, H_True.T), 
                       fig_size=[20,10], 
                       show_importance=True)

Display_Class.display_dict_and_graph(title='W_False-20-walks-WS-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)



        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'WS_GHM_Dict_to_Graph_Deg')



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



c_True = df_True.loc[:, 'E_0_0':'E_19_19']
c_False = df_False.loc[:,'E_0_0':'E_19_19']
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
Display_Class.display_dict_and_graph(title='W_True-20-walks-UCLA-importance',
                       W=W_True, 
                       At = np.dot(H_True, H_True.T), 
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-20-walks-UCLA-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)



        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'UCLA26_GHM_Dict_to_Graph_Deg')


"""
Harvard Graph GHM
"""
WSD = pd.read_csv('GHM_Dict_Data_Harvard1_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title5 = 'Adj_Matrix for Harvard1_GHM Sync'
title6 = 'Adj_Matrix for Harvard1_GHM Non-Sync'
# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:, 'E_0_0':'E_19_19']
c_False = df_False.loc[:,'E_0_0':'E_19_19']
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
Display_Class.display_dictionary(title5, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title6, X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-20-walks-Harvard-importance',
                       W=W_True, 
                       At = np.dot(H_True,H_True.T),
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-20-walks-Harvard-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)




        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'Harvard1_GHM_Dict_to_Graph_Deg')


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
title7 = 'Adj_Matrix for Caltech36_GHM Sync'
title8 = 'Adj_Matrix for Caltech36_GHM Non-Sync'
# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:, 'E_0_0':'E_19_19']
c_False = df_False.loc[:,'E_0_0':'E_19_19']
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
Display_Class.display_dictionary(title7, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title8, X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-20-walks-Caltech-importance',
                       W=W_True, 
                       At = np.dot(H_True,H_True.T),
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-20-walks-Caltech-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)




        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'Caltech36_GHM_Dict_to_Graph_Deg')


"""
Wisconsin87 Graph GHM
"""
WSD = pd.read_csv('GHM_Dict_Data_Wisconsin87_AllStates.csv')
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())
y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
title9 = 'Adj_Matrix for Wisconsin87_GHM Sync'
title10 = 'Adj_Matrix for Wisconsin87_GHM Non-Sync'
# Edge only
# c_True = df_True.loc[:,'':'']



c_True = df_True.loc[:, 'E_0_0':'E_19_19']
c_False = df_False.loc[:,'E_0_0':'E_19_19']
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
Display_Class.display_dictionary(title9, W=X_True[:,:100], figsize=[20,20])
Display_Class.display_dictionary(title10, X_False[:,:100], figsize=[20,20])
Display_Class.display_dict_and_graph(title='W_True-20-walks-Wisconsin-importance',
                       W=W_True, 
                       At = np.dot(H_True,H_True.T),
                       fig_size=[20,10], 
                       show_importance=True)
Display_Class.display_dict_and_graph(title='W_False-20-walks-Wisconsin-importance',
                       W=W_False, 
                       At = np.dot(H_False, H_False.T), 
                       fig_size=[20,10], 
                       show_importance=True)




        
n=8        
Display_Class.plot_adj_to_graph_deg(W_True, W_False, n, True, title = 'Wisconsin87_GHM_Dict_to_Graph_Deg')
