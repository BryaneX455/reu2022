# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:55:39 2022

@author: Bryan
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from NNetwork_master.src.NNetwork import NNetwork as nn
from GHM_Classes.GHM import GHM

# UCLA26
Kap = 5
ItNum = 25
G_Num = 10
def generate_nxg(X):

    graph_list = []
    k = int(np.sqrt(X.shape[0]))
    for i in range(X.shape[1]):
        adj_mat = X.T[i].reshape(k,k)
        G = nx.from_numpy_matrix(adj_mat)
        graph_list.append(G)
    
    return graph_list

def GHM_Adj_Color_Diff(G_Num, Graph_List, Node_Num, Kap, ItNum, title):
    fig, axs = plt.subplots(8,G_Num,figsize = (25,15))
    BaseName = 'Colored Adj Matrix'
    Full_Mat_AllG = []
    N = 0
    SyncList = []
    for G in Graph_List:
        s = np.random.randint(5, size=1*Node_Num)
        if nx.is_connected(G):
            Number_Nodes = G.number_of_nodes()
            Adj_Mat = nx.to_numpy_matrix(G).tolist()
            Full_Mat = list(np.empty((ItNum+1, Number_Nodes, Number_Nodes)))
            GHM_Class = GHM(G, s, Kap, ItNum)
            st, sync = GHM_Class.GHM1D()
            for k in range(ItNum+1):
                for i in range(Number_Nodes):
                    for j in range(Number_Nodes):
                        if k==0:
                            Full_Mat[k][i][j] = int(Adj_Mat[i][j])
                        else:
                            Color_Diff = int(min([abs(st[k-1][i] - st[k-1][j]), 5 - abs(st[k-1][i] - st[k-1][j])]))
                            Full_Mat[k][i][j] = Color_Diff
            Full_Mat_AllG.append(Full_Mat)
            if sync:
                SyncList.append(1)
            else:
                SyncList.append(0)

            sb.heatmap(Full_Mat[0][:][:], ax = axs[0,N])
            sb.heatmap(Full_Mat[1][:][:], ax = axs[1,N])
            sb.heatmap(Full_Mat[2][:][:], ax = axs[2,N])
            sb.heatmap(Full_Mat[3][:][:], ax = axs[3,N])
            sb.heatmap(Full_Mat[4][:][:], ax = axs[4,N])
            sb.heatmap(Full_Mat[5][:][:], ax = axs[5,N])
            sb.heatmap(Full_Mat[10][:][:], ax = axs[6,N])
            sb.heatmap(Full_Mat[25][:][:], ax = axs[7,N])
        N+=1
    fig.suptitle(title+BaseName, fontsize = 40)
    plt.show()
    fig.subplots_adjust(wspace=0.5)
    fig.subplots_adjust(hspace=0.5)
    
    return Full_Mat_AllG, SyncList

sampling_alg = 'pivot'



# UCLA26
ntwk = 'UCLA26' # COVID_PPI, Wisconsin87, Caltech36
ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
k=50

path = str(ntwk) + '.txt'
G = nn.NNetwork()
G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)


X, embs = G.get_patches(k=k, sample_size=G_Num, skip_folded_hom=True)
graph_list = generate_nxg(X)

Color_Diff_Mat, SyncList = GHM_Adj_Color_Diff(G_Num, graph_list, k, Kap, ItNum, ntwk)
print(len(Color_Diff_Mat))
for i in range(10):   
    Color_Diff_12 = np.hstack((Color_Diff_Mat[0],Color_Diff_Mat[1]))
    Color_Diff_34 = np.hstack((Color_Diff_Mat[2],Color_Diff_Mat[3]))
    Color_Diff_1234 = np.vstack((Color_Diff_12,Color_Diff_34))


# Caltech36
ntwk = 'Caltech36' # COVID_PPI, Wisconsin87, Caltech36
ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
k=50

path = str(ntwk) + '.txt'
G = nn.NNetwork()
G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
#print('num nodes in G', len(G.nodes()))
#print('num edges in G', len(G.get_edges()))

X, embs = G.get_patches(k=k, sample_size=G_Num, skip_folded_hom=True)
graph_list = generate_nxg(X)

GHM_Adj_Color_Diff(G_Num, graph_list, k, Kap, ItNum, ntwk)

