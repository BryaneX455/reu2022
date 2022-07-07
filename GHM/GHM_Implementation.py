# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:55:03 2022

@author: Zihong Xu
"""



from NNetwork.NNetwork import NNetwork as NNT
import numpy as np
import pandas   as pd
import seaborn as sb

""" Parameter: Graph G, Phase State S, Period k,Iteration Number ItNum """
def GHM(G,S,k,ItNum):
    St = S
    SN = S
    for i in range(ItNum):
        if i!=0:
            S = SN
            St = np.vstack((St,SN))
        for j in range(G.number_nodes):
            Onein = False
            NeighbNum = len(G.neighbors(G.vertices[j]))
            NeighbSet = G.neighbors(G.vertices[j])
            NeighbList = list(NeighbSet)
            print(NeighbNum)
            print(NeighbList)
            NeighbState = np.zeros(NeighbNum)
            for k in range(NeighbNum):
                NeighbState[k] = S[int(NeighbList[k])-1]  
            print(NeighbState)    
            print(SN)
            if 1 in NeighbState:
                Onein = True
            print(Onein)
            if SN[j] == 0 and (not Onein):
                SN[j] = 0
            elif SN[j] == 0 and Onein:
                SN[j] = 1
            else:
                SN[j] = (SN[j] + 1)% k
            print(SN[j])
    
    return St     
   
        
        
"""Non-Synchronizing example"""
edgelist = [['1','2'],['2','3'],['3','4'],['1','3'],['1','5'],['1','6'],['1','7'],['2','4'],['2','5'],['3','6'],\
            ['3','4'],['3','5'],['3','6'],['4','5'],['4','6'],['4','7'],['5','6'],['5','7']]
G1 = NNT()
G1.add_edges(edgelist)

State = GHM(G1,[0,2,1,3,2,1,2],5,10)
G1map = pd.DataFrame(State)
sb.heatmap(G1map)

"""Collapsing Example"""
