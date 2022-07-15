# -*- coding: utf-8 -*-
"""
Created on Tue Jimport networkx as nx
@author: Bryan
"""
import random
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier



def animate_GHM2D(G,S,Kap,ItNum):
    (m,n) = S.shape  
    fig = plt.figure()
    def init():
        sb.heatmap(S,cbar=False, cmap='viridis')
    
    def animate(i):
        PhaseStatei = PhaseStateList[i]
        sb.heatmap(PhaseStatei,cbar=False, cmap='viridis')
        
    PhaseStateList = [[[0 for k in range(n)] for j in range(m)] for i in range(ItNum)]
    
        
    NodeNum = G.number_of_nodes()
           
    SN = list(np.zeros(S.shape))
    for i in range(m):
        for j in range(n):
            SN[i][j] = S[i][j]
    for i in range(ItNum):
        if i!=0:
            for l1 in range(m):
                for l2 in range(n):
                    S[l1][l2] = SN[l1][l2]
        for j in range(NodeNum):
            Onein = False
            NeighbSet2D = G.neighbors(list(G.nodes)[j])
            NeighbList2D = list(NeighbSet2D)
            NeighbNum2D = len(NeighbList2D)
            ProsNeighbNum = 1
            NeighbInd = random.sample(range(NeighbNum2D),ProsNeighbNum)
            TheNeighbs = [NeighbList2D[NeighbInd[y]] for y in range(ProsNeighbNum)]
            NeighbState = [S[TheNeighbs[x][0]][TheNeighbs[x][1]] for x in range(ProsNeighbNum)]

            if 1 in NeighbState:
                Onein = True
            if S[int((j-j%n)/n)][int(j%n)] == 0 and (not Onein):
                R1 = random.uniform(0,1)
                if R1 <= 1:
                    SN[int((j-j%n)/n)][int(j%n)] = 0
                    
            elif S[int((j-j%n)/n)][int(j%n)] == 0 and Onein:
                R2 = random.uniform(0, 1)
                if R2 <= 1:
                    SN[int((j-j%n)/n)][int(j%n)] = 1
                
            else:
                SN[int((j-j%n)/n)][int(j%n)] = (SN[int((j-j%n)/n)][int(j%n)] + 1) % Kap
    

        for j in range(m):
            for k in range(n):                
                PhaseStateList[i][j][k] = S[j][k]
                           
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=ItNum, repeat = False)

    FileName = 'GHM2DAnimation.gif'
    pillowwriter = animation.PillowWriter(fps=1)
    anim.save(FileName, writer=pillowwriter)
    return PhaseStateList

TestNum = 20
SyncNum = 0    
for l2 in range(TestNum):
    a = 50
    b = 50
    kap = 5
    ItNum = 30
    Grand2D = nx.grid_2d_graph(a,b) 
    s = np.random.randint(5, size=(a,b))
    PhaseStateL = animate_GHM2D(Grand2D,s,kap,ItNum)
    FinalState = PhaseStateL[ItNum-1]
    if not np.any(FinalState):
        SyncNum += 1
print(SyncNum)




        



