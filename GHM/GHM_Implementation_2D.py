# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 23:11:38 2022

@author: 16088
"""


import networkx as nx

import time
import random
import numpy as np
import pandas as pd
from IPython import display
from IPython.core.display import HTML
import seaborn as sb
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
        sb.heatmap(S,cbar=False, cmap='viridis')

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
    St = SArr
    SN = np.zeros(G.number_of_nodes())
    NodeNum = G.number_of_nodes()
    for i in range(len(S)):
        SN[i] = SArr[i]            
            
    for i in range(ItNum):
        PhaseState = np.zeros(S.shape)

        if i!=0:
            SArr = SN
            St = np.vstack((St,SN))
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
                l = random.uniform(0, 1)
                if l <= 0.8:
                    SN[j] = 0
                else:
                    SN[j] = SN[j]
            elif SArr[j] == 0 and Onein:
                l = random.uniform(0, 1)
                if l <= 0.8:
                    SN[j] = 1 % Kap
                else:
                    SN[j] = SN[j]
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

    FileName = "GHM2D_Animation.gif"
    pillowwriter = animation.PillowWriter(fps=20)
    anim.save(FileName, writer=pillowwriter)
    
    plt.show()

a = 20
b = 35
Grand2D = nx.grid_2d_graph(a,b) 
s = np.random.randint(7, size=(a,b))
animate_GHM2D(Grand2D,s,7,20)

imageObject = Image.open("GHM2D_Animation.gif")

print(imageObject.is_animated)

print(imageObject.n_frames)

 

# Display individual frames from the loaded animated GIF file

for frame in range(0,imageObject.n_frames):

    imageObject.seek(frame)

    imageObject.show()




