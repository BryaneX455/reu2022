# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:28:35 2022

@author: Bryan
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sb
import networkx as nx

class GHM_Animation:
    def __init__(self, G = None, S = None, Kap = None, ItNum = None):
        
        self.G = G
        self.S = S
        self.Kap = Kap
        self.ItNum = ItNum
        
    def Animate_GHM2D(self):
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