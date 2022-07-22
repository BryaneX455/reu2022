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

from itertools import product



graph = nx.watts_strogatz_graph(12,4,0.7)

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
            NeighbNum = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet = G.neighbors(list(G.nodes)[j])
            NeighbList = list(NeighbSet)
            NeighbState = list(np.zeros(NeighbNum))
            for k in range(NeighbNum):
                NeighbState[k] = S[k]
                

            if 1 in NeighbState:
                Onein = True
            if S[j] == 0 & (not Onein):
                SN[j] = 0
            elif S[j] == 0 & Onein:
                SN[j] = 1 % kap
            else:
                SN[j] = (SN[j] + 1) % kap
    Sync = False            
    if np.all((St[ItNum-1] == 0)):
        Sync = True
    PhaseState = pd.DataFrame(St)
    sb.heatmap(PhaseState, cbar=False, cmap='viridis') 
    return St, Sync 

NodeNum = 10
kap = 5
knn = 4
#s = np.random.randint(kap, size=NodeNum)
color_list = list(product(range(0, kap), repeat=NodeNum))
print(len(color_list))
file = open('GHM.csv', 'w+', newline ='')
GHMItNum = 30
SyncNum = 0
Non_Sync_Num = 0
G = nx.watts_strogatz_graph(NodeNum,knn,0.65)
for col in color_list:
    states, label = GHM(G, col, kap, GHMItNum)
    print(label)
    if label:
        SyncNum += 1
    else:
        Non_Sync_Num += 1
print(SyncNum)
print(Non_Sync_Num)
