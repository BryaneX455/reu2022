# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:48:06 2022

@author: Bryan
"""

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sb


class GHM:
    
    def __init__(self, G = None, S = None, Kap = None, ItNum = None):
        
        self.G = G
        self.S = S
        self.Kap = Kap
        self.ItNum = ItNum
        
    
    def GHM1D(self):
        S = list(self.S)
        St = np.zeros(len(S))
        SN = np.zeros(len(S))
        for s1 in range(len(S)):
            St[s1] = S[s1]
            SN[s1] = S[s1]
        for i in range(self.ItNum):
            if i!=0:
                for s2 in range(len(S)):
                    S[s2] = SN[s2]
                St = np.vstack((St,SN))
            for j in range(self.G.number_of_nodes()):
                Onein = False
                StateZero = False
                NeighbNum = len(list(self.G.neighbors(list(self.G.nodes)[j])))
                NeighbSet = self.G.neighbors(list(self.G.nodes)[j])
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
                    SN[j] = 1 % self.Kap
                else:
                    SN[j] = (SN[j] + 1) % self.Kap
        Sync = False           
        if np.all((St[self.ItNum-1] == 0)):
            Sync = True
        PhaseState = pd.DataFrame(St)
        sb.heatmap(PhaseState, cbar=False, cmap='viridis') 
        
        return St, Sync  
    
    def animate_GHM2D(self):
        """Implements 2D Greenberg Hasting Model Animation
    
        Args:
            G (NetworkX Graph): Input graph to the model
            S (array): Current state
            Kap (int): Kap-color GHM
            ItNum (int): Number of iterations
    
        Returns:
            2D heatmap animation for the GHM
        """
        S = self.S
        (m,n) = S.shape  
        fig = plt.figure()
        def init():
            sb.heatmap(S,cbar=False, cmap='viridis')
        
        def animate(i):
            PhaseStatei = PhaseStateList[i]
            sb.heatmap(PhaseStatei,cbar=False, cmap='viridis')
            
        PhaseStateList = [[[0 for k in range(n)] for j in range(m)] for i in range(self.ItNum)]
        
            
        NodeNum = self.G.number_of_nodes()
               
        SN = list(np.zeros(S.shape))
        for i in range(m):
            for j in range(n):
                SN[i][j] = S[i][j]
        for i in range(self.ItNum):
            if i!=0:
                for l1 in range(m):
                    for l2 in range(n):
                        S[l1][l2] = SN[l1][l2]
            for j in range(NodeNum):
                Onein = False
                NeighbNum2D = len(list(self.G.neighbors(list(self.G.nodes)[j])))
                NeighbSet2D = self.G.neighbors(list(self.G.nodes)[j])
                NeighbList2D = list(NeighbSet2D)
                NeighbState = list(np.zeros(NeighbNum2D))
                
                for k in range(NeighbNum2D):
                    NeighbState[k] = S[NeighbList2D[k][0]][NeighbList2D[k][1]]
                if 1 in NeighbState:
                    Onein = True
                if S[int((j-j%n)/n)][int(j%n)] == 0 and (not Onein):
                    SN[int((j-j%n)/n)][int(j%n)] = 0
                elif S[int((j-j%n)/n)][int(j%n)] == 0 and Onein:
                    SN[int((j-j%n)/n)][int(j%n)] = 1
                else:
                    SN[int((j-j%n)/n)][int(j%n)] = (SN[int((j-j%n)/n)][int(j%n)] + 1) % self.Kap
        
        
            print(S)
    
            for j in range(m):
                for k in range(n):                
                    PhaseStateList[i][j][k] = S[j][k]
                               
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=self.ItNum, repeat = False)
    
        FileName = 'GHM2DAnimation.gif'
        pillowwriter = animation.PillowWriter(fps=1)
        anim.save(FileName, writer=pillowwriter)
        
        plt.show()    
    
    
    def GHMGridToArr(self):
        """Implements Vectorized 2D Grennberg Hasting Model 
    
        Args:
            G (NetworkX Graph): Input graph to the model
            S (array): Current state
            Kap (int): Kap-color GHM
            ItNum (int): Number of iterations
    
        Returns:
            Generates heatmap of time evolution of vectorized 2D GHM model
        """
        
        GArr = list(np.zeros(self.G.number_of_nodes()))
        ColNum = 0
        for i in range(self.G.number_of_nodes()):
            if list(self.G.nodes)[i][0] == 0:
                ColNum += 1
        RowNum = int(self.G.number_of_nodes()/ColNum)
        for i in range(RowNum):
            for j in range(ColNum):
                GArr[int(i)*ColNum + int(j)] = int(i)*ColNum + int(j)
        St = self.S
        S_Temp = np.zeros(self.G.number_of_nodes())
        SN = np.zeros(self.G.number_of_nodes())
        NodeNum = self.G.number_of_nodes()
        for i in range(len(self.S)):
            SN[i] = self.S[i]
        for i in range(len(self.S)):
            S_Temp[i] = self.S[i]
        for i in range(self.ItNum):
            if i!=0:
                for l1 in range(len(self.S)):
                    S_Temp[l1] = SN[l1]
                St = np.vstack((St,SN))
            for j in range(NodeNum):
                Onein = False
                NeighbNum2D = len(list(self.G.neighbors(list(self.G.nodes)[j])))
                NeighbSet2D = self.G.neighbors(list(self.G.nodes)[j])
                NeighbList2D = list(NeighbSet2D)
                NeighbPos = list(np.zeros(len(NeighbList2D)))
                for k in range(len(NeighbList2D)):
                    NeighbPos[k] = NeighbList2D[k][0]*ColNum+NeighbList2D[k][1]
                NeighbState2D = np.zeros(NeighbNum2D)
                for k in range(NeighbNum2D):
                    NeighbState2D[k] = S_Temp[int(NeighbPos[k])]  
    
                if 1 in NeighbState2D:
                    Onein = True
                if S_Temp[j] == 0 and (not Onein):
                    SN[j] = 0
                elif S_Temp[j] == 0 and Onein:
                    SN[j] = 1 % self.Kap
                else:
                    if (S_Temp[j] + 1) % self.Kap == 0:
                        
                        SN[j]=0
                    else:       
                        SN[j] = (SN[j] + 1) % self.Kap
        
        PhaseState = pd.DataFrame(St)
        
    
        return sb.heatmap(PhaseState, cbar=False, cmap='viridis'), St 

