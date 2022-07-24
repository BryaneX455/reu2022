# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:58:14 2022
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

NodeNum = 14
kap = 5
knn = 4
prob = 0.65
GHMItNum = 30


def sample_WS(num_samples, NodeNum, knn, prob, kap, GHMItNum):
    df = pd.DataFrame(columns=['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
    
    for i in range(num_samples):
        
        G = nx.watts_strogatz_graph(NodeNum, knn, prob)
        s = np.random.randint(5, size=1*NodeNum)
        if nx.is_connected(G):
            # Number of Edges and Nodes
            edges = G.number_of_edges()
            nodes = G.number_of_nodes()

            # Min and Max degree
            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)

            # Diameter of graph
            diam = nx.diameter(G)
            
            # Applying GHM
            adj_mat = nx.to_numpy_array(G)
            
            state, label = GHM(G, s, kap, GHMItNum)
            if label:
                label = 1
            else:
                label = 0
            df.at[len(df.index)] = [kap, edges, nodes, dmin, dmax, diam, label]
    
    return df

SampleNum = 700
df =  sample_WS(SampleNum, NodeNum, knn, prob, kap, GHMItNum)
df.to_csv('GHM.csv')  
