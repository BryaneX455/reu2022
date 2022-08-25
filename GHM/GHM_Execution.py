# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:49:24 2022

@author: 16088
"""
import networkx as nx
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from GHM_Classes.GHM import GHM
from GHM_Classes.GHM_Animation import GHM_Animation



NodeNum = 20
knn = 4
prob= 0.65
Kap = 5
ItNum = 30


# Watts Strogatz 1-D
G = nx.watts_strogatz_graph(NodeNum, knn, prob)
s = np.random.randint(5, size=1*NodeNum)
GHM_Class = GHM(G, s, Kap, ItNum)
st, sync = GHM_Class.GHM1D()
sb.heatmap(st)
plt.title('GHM1D_WS_20Nodes_4knn_0.65Rewiring')
plt.show()

# Networkx 2-D grid
G = nx.grid_2d_graph(8, 8) 
s = np.random.randint(5, size=1*64)
GHM_Class = GHM(G, s, Kap, ItNum)
GHM_Class.GHMGridToArr()
plt.title('GHM2D_64Nodes_Sample')
plt.show()

# Networkx 2-D grid Animastion
G = nx.grid_2d_graph(60, 60) 
s = np.random.randint(5, size=(60,60))
GHM_Animation_Class = GHM_Animation(G, s, Kap, ItNum)
GHM_Animation_Class.Animate_GHM2D()