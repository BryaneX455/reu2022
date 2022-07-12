# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 23:11:38 2022

@author: 16088
"""


import networkx as nx

import random
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
from PIL import GifImagePlugin


def animate_GHM2D(G,S,Kap,ItNum):
    (m,n) = S.shape  
    fig = plt.figure()
    def init():
        sb.heatmap(S,cbar=False, cmap='viridis')
    
    def animate(i):
        PhaseStatei = PhaseStateList[i]
        sb.heatmap(PhaseStatei,cbar=False, cmap='viridis')
        
    PhaseStateList = [[[0 for k in range(n)] for j in range(m)] for i in range(ItNum)]
    print(np.array(PhaseStateList).shape)
    
        
    GArr = list(np.zeros(G.number_of_nodes()))
    for i in range(m):
        for j in range(n):
            GArr[int(i)*n + int(j)] = int(i)*n + int(j)
    SArr = list(np.zeros(m*n))
    for i in range(m*n):
        ColPos = int(i % n)
        RowPos = int((i - ColPos)/n)
        SArr[i] = S[RowPos][ColPos]
    SN = np.zeros(G.number_of_nodes())
    NodeNum = G.number_of_nodes()

    for i in range(NodeNum):
        SN[i] = SArr[i]            
            
    for i in range(ItNum):
        PhaseState = np.zeros(S.shape)

        if i!=0:
            SArr = SN
        for j in range(NodeNum):
            Onein = False
            NeighbNum2D = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet2D = G.neighbors(list(G.nodes)[j])
            NeighbList2D = list(NeighbSet2D)
            NeighbPos = list(np.zeros(len(NeighbList2D)))
            for k in range(len(NeighbList2D)):
                NeighbPos[k] = NeighbList2D[k][0]*n+NeighbList2D[k][1]
            NeighbState2D = np.zeros(NeighbNum2D)
        
            for k in range(NeighbNum2D):
                NeighbState2D[k] = SArr[int(NeighbPos[k])]  

            if 1 in NeighbState2D:
                Onein = True
            if SArr[j] == 0 and (not Onein):
                SN[j] = 0
            elif SArr[j] == 0 and Onein:
                SN[j] = 1
            else:
                if (SArr[j] + 1) % Kap == 0:
                    
                    SN[j]=0
                else:       
                    SN[j] = (SN[j] + 1) % Kap
    
    
        for p in range(m):
            for q in range(n):
                
                PhaseState[p][q] = SArr[p*n + q]
                    
                
        for j in range(m):
            for k in range(n):                
                PhaseStateList[i][j][k] = PhaseState[j][k]
                
    print(PhaseStateList)            
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=ItNum, repeat = False)

    #FileName = "GHM2D_Animation.mp4"
    #FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
    FileName = 'GHM2DAnimation.gif'
    pillowwriter = animation.PillowWriter(fps=1)
    anim.save(FileName, writer=pillowwriter)
    
    plt.show()
    
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

a = 60
b = 60
kap = 5
ItNum = 45
Grand2D = nx.grid_2d_graph(a,b) 
s = np.random.randint(10, size=(a,b))
SArr = list(np.zeros(a*b))
for i in range(a*b):
    ColPosi = int(i % b)
    RowPosi = int((i-ColPosi)/b)
    SArr[i] = s[RowPosi][ColPosi]
animate_GHM2D(Grand2D,s,kap,ItNum)
plt.figure(2)
GHMGridToArr(Grand2D,SArr,kap,ItNum)

