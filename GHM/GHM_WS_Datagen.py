# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:58:14 2022
@author: Zihong Xu
"""
import math
import numpy as np
import pandas   as pd
import seaborn as sb
import networkx as nx
import statistics as st
import matplotlib.pyplot as plt
import random
import csv

from pathlib import Path  
from itertools import product


def GHM(G,S,kap,ItNum):
    S = list(S)
    St = np.zeros(len(S))
    SN = np.zeros(len(S))
    for s1 in range(len(S)):
        St[s1] = S[s1]
        SN[s1] = S[s1]
    for i in range(ItNum):
        if i!=0:
            for s2 in range(len(S)):
                S[s2] = SN[s2]
            St = np.vstack((St,SN))
        for j in range(G.number_of_nodes()):
            Onein = False
            StateZero = False
            NeighbNum = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet = G.neighbors(list(G.nodes)[j])
            NeighbList = list(NeighbSet)
            NeighbState = list(np.zeros(NeighbNum))
            for k in range(NeighbNum):
                NeighbState[k] = S[int(NeighbList[k])-1]
                

            if 1 in NeighbState:
                Onein = True
            if S[j] == 0:
                StateZero = True
            if StateZero & (not Onein):
                SN[j] = 0
            elif StateZero & Onein:
                SN[j] = 1 % kap
            else:
                SN[j] = (SN[j] + 1) % kap
    Sync = False           
    if np.all((St[ItNum-1] == 0)):
        Sync = True
    PhaseState = pd.DataFrame(St)
    sb.heatmap(PhaseState, cbar=False, cmap='viridis') 
    
    return St, Sync 

NodeNum = 16
kap = 5
knn = 4
prob = 0.65
GHMItNum = 30


def sample_WS(num_samples, NodeNum, prob, knn, kap, GHMItNum):
    BaseName = "S" 
    Edge_Base_Name = "E"
    InitPhaseList = list(np.zeros(NodeNum))
    EdgeList = list(np.zeros(int(NodeNum*NodeNum)))
    for i in range(int(NodeNum*NodeNum)):
        Row_Pos = math.floor(i/NodeNum)
        Col_Pos = i%NodeNum
        Row_Pos_Name = str(Row_Pos)
        Col_Pos_Name = str(Col_Pos)
        EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'

    for i in range(NodeNum):
        InitPhaseList[i] = f'{BaseName}_{i}_{1}'
    ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
    ColList.extend(InitPhaseList)
    ColList.extend(EdgeList)
    ColList.extend(['label'])
    print(ColList)
    df = pd.DataFrame(columns=ColList)
    
    SyncNum = 0
    NonSync = 0
    for i in range(num_samples):
        print(i)
        G = nx.watts_strogatz_graph(NodeNum, knn, prob)
        s = np.random.randint(5, size=1*NodeNum)
        if nx.is_connected(G):
            # Number of Edges and Nodes
            # EdgeList = G.edges
            edges = G.number_of_edges()
            nodes = G.number_of_nodes()

            # Min and Max degree
            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)

            # Diameter of graph
            diam = nx.diameter(G)
            
            # Applying GHM
            Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
            Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
            state, label = GHM(G, s, kap, GHMItNum)
            if label:
                label = 1
                SyncNum += 1
            else:
                label = 0
                NonSync += 1
            
            SList = list(s)
            SList.extend(Vec_Adj_Matrix)
            SList.append(label)
            if max(SyncNum,NonSync) <= 1225:
                df.at[len(df.index)] = [kap, edges, nodes, dmin, dmax, diam]+SList
                
            elif min(SyncNum,NonSync) <= 1225:
                if min(SyncNum,NonSync) == SyncNum:
                    if label == 1:
                        df.at[len(df.index)] = [kap, edges, nodes, dmin, dmax, diam]+SList
                elif min(SyncNum,NonSync) == NonSync:
                    if label == 0:
                        df.at[len(df.index)] = [kap, edges, nodes, dmin, dmax, diam]+SList
    

    return df, SyncNum

SampleNum = 3000
df,SyncNum =  sample_WS(SampleNum, NodeNum, prob, knn, kap, GHMItNum)
df.to_csv('GHM_Dict_Data.csv')  
print(SyncNum)



