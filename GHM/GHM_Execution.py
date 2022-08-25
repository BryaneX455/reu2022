# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:49:24 2022

@author: 16088
"""
import networkx as nx
import numpy as np
import seaborn as sb
from GHM_Classes.GHM import GHM

NodeNum = 20
knn = 4
prob= 0.65
Kap = 5
ItNum = 30
G = nx.watts_strogatz_graph(NodeNum, knn, prob)
s = np.random.randint(5, size=1*NodeNum)
GHM_Class = GHM(G, s, Kap, ItNum)
st, sync = GHM_Class.GHM1D()
sb.heatmap(st)