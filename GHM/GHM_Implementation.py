# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:55:03 2022

@author: Zihong Xu
"""




import numpy as np
import pandas   as pd
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
import random

""" Parameter: Graph G, Phase State S, Period k,Iteration Number ItNum """
def GHMArr(G,S,kap,ItNum):
    St = S
    SN = np.zeros(G.number_of_nodes())
    NodeNum = G.number_of_nodes()
    for i in range(len(S)):
        SN[i] = S[i]
    for i in range(ItNum):
        if i!=0:
            S = SN
            St = np.vstack((St,SN))
        for j in range(NodeNum):
            Onein = False
            NeighbNum = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet = G.neighbors(list(G.nodes)[j])
            NeighbList = list(NeighbSet)
            NeighbState = np.zeros(NeighbNum)
            for k in range(NeighbNum):
                NeighbState[k] = S[int(NeighbList[k])-1]  

            if 1 in NeighbState:
                Onein = True
            if S[j] == 0 and (not Onein):
                SN[j] = 0
            elif S[j] == 0 and Onein:
                SN[j] = 1 % kap
            else:
                if (S[j] + 1) % kap == 0:
                    
                    SN[j]=0
                else:       
                    SN[j] = (SN[j] + 1) % kap
    
    PhaseState = pd.DataFrame(St)
    

    return sb.heatmap(PhaseState, cbar=False, cmap='viridis')     
   
        
        
"""Synchronizing example"""
edgelist = [['1','2'],['2','3'],['3','4'],['1','3'],['1','5'],['1','6'],['1','7'],['1','8'],['2','8'],['2','9'],['1','9'],['2','4'],['2','5'],['3','6'],\
            ['3','4'],['3','5'],['3','6'],['3','8'],['3','9'],['4','5'],['4','6'],['4','7'],['4','9'],['5','6'],['5','7'],['6','8']]
G1 = nx.Graph()
G1.add_edges_from(edgelist)
plt.figure(1)
nx.draw(G1, with_labels=True, font_weight='bold')
plt.figure(2)
s = np.random.randint(7,size = 1*9)
GHMArr(G1, s, 12, 15);


def GHMGridToArr(G,S,kap,ItNum):
    GArr = list(np.zeros(G.number_of_nodes()))
    ColNum = 0
    for i in range(G.number_of_nodes()):
        if list(G.nodes)[i][0] == 0:
            ColNum += 1
    RowNum = int(G.number_of_nodes()/ColNum)
    for i in range(RowNum):
        for j in range(ColNum):
            GArr[int(i)*ColNum + int(j)] = int(i)*ColNum + int(j)
    St = S
    SN = np.zeros(G.number_of_nodes())
    NodeNum = G.number_of_nodes()
    for i in range(len(S)):
        SN[i] = S[i]
    for i in range(ItNum):
        if i!=0:
            S = SN
            St = np.vstack((St,SN))
        for j in range(NodeNum):
            Onein = False
            NeighbNum2D = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet2D = G.neighbors(list(G.nodes)[j])
            NeighbList2D = list(NeighbSet2D)
            NeighbPos = list(np.zeros(len(NeighbList2D)))
            for k in range(len(NeighbList2D)):
                NeighbPos[k] = NeighbList2D[k][0]*ColNum+NeighbList2D[k][1]
            NeighbState2D = np.zeros(NeighbNum2D)
            for k in range(NeighbNum2D):
                NeighbState2D[k] = S[int(NeighbPos[k])]  

            if 1 in NeighbState2D:
                Onein = True
            if S[j] == 0 and (not Onein):
                SN[j] = 0
            elif S[j] == 0 and Onein:
                SN[j] = 1 % kap
            else:
                if (S[j] + 1) % kap == 0:
                    
                    SN[j]=0
                else:       
                    SN[j] = (SN[j] + 1) % kap
    
    PhaseState = pd.DataFrame(St)
    

    return sb.heatmap(PhaseState, cbar=False, cmap='viridis'), St  

def GHMGridToArrRand(G,S,kap,ItNum):
    GArr = list(np.zeros(G.number_of_nodes()))
    ColNum = 0
    for i in range(G.number_of_nodes()):
        if list(G.nodes)[i][0] == 0:
            ColNum += 1
    RowNum = int(G.number_of_nodes()/ColNum)
    for i in range(RowNum):
        for j in range(ColNum):
            GArr[int(i)*ColNum + int(j)] = int(i)*ColNum + int(j)
    St = S
    SN = np.zeros(G.number_of_nodes())
    NodeNum = G.number_of_nodes()
    for i in range(len(S)):
        SN[i] = S[i]
    for i in range(ItNum):
        if i!=0:
            S = SN
            St = np.vstack((St,SN))
        for j in range(NodeNum):
            Onein = False
            NeighbNum2D = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet2D = G.neighbors(list(G.nodes)[j])
            NeighbList2D = list(NeighbSet2D)
            NeighbPos = list(np.zeros(len(NeighbList2D)))
            for k in range(len(NeighbList2D)):
                NeighbPos[k] = NeighbList2D[k][0]*ColNum+NeighbList2D[k][1]
            NeighbState2D = np.zeros(NeighbNum2D)
            
            for k in range(NeighbNum2D):
                NeighbState2D[k] = S[int(NeighbPos[k])]  

            if 1 in NeighbState2D:
                Onein = True
            if S[j] == 0 and (not Onein):
                l = random.uniform(0, 1)
                if l <= 0.7:
                    SN[j] = 0
                else:
                    SN[j] = SN[j]
            elif S[j] == 0 and Onein:
                l = random.uniform(0, 1)
                if l <= 0.7:
                    SN[j] = 1 % kap
                else:
                    SN[j] = SN[j]
            else:
                if (S[j] + 1) % kap == 0:
                    
                    SN[j]=0
                else:       
                    SN[j] = (SN[j] + 1) % kap
    
    PhaseState = pd.DataFrame(St)
    

    return sb.heatmap(PhaseState, cbar=False, cmap='viridis'), St

RandWin = 0
NormalWin = 0
TransRandSuccessNum = 0
TieNum = 0
for a in range(5):
   for i in range(10):  
       a = 15
       b = 14
       Grand2DArr = nx.grid_2d_graph(a,b) # When the b is larger than a, there is an index bug!
       s = np.random.randint(7, size=1*(a*b))
       plt.figure(3)
       MapNorm, StNorm = GHMGridToArr(Grand2DArr,s,7,30)
       plt.figure(4)
       MapRand, StRand = GHMGridToArrRand(Grand2DArr,s,7,30)
       NormSync = False
       RandSync = False
       if all([ Sn == 0 for Sn in StNorm[29] ]):
           NormSync = True
       if all([ Sn == 0 for Sn in StRand[29] ]):
           RandSync = True
       if (not NormSync) and RandSync:
           RandWin += 1
       elif NormSync and (not RandSync):
           NormSync += 1        
   if RandWin > NormalWin:
       TransRandSuccessNum += 1
       print('Transition Rule Success')
   elif RandWin == NormalWin:
       TieNum += 1
       print('Tie')

print(TransRandSuccessNum)
print(TieNum)
