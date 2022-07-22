# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 23:34:29 2022

@author: Bryan
"""

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sb
import random

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

def sample_WS(num_samples, nodes, neighbors, probability, half_sync=False):
    df = pd.DataFrame(columns=['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
    
    for i in range(num_samples):
        
        G = nx.watts_strogatz_graph(nodes, neighbors, probability)
        kappa = random.uniform(0.5, 4.5)
        
        if nx.is_connected(G):
            # Number of Edges and Nodes
            edges = G.number_of_edges()
            nodes = G.number_of_nodes()

            # Min and Max degree
            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)

            # Diameter of graph
            diam = nx.diameter(G)
            
            # Applying Kuramoto
            adj_mat = nx.to_numpy_array(G)
            
            if half_sync:
                natfreqs = np.repeat(2 * np.pi * 0, nodes)
                model = GHM(coupling=kappa, dt=0.01, T=50, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = GHM(coupling=kappa, dt=0.01, T=50, n_nodes=nodes, half_sync=half_sync)
            
            sim = model.run(adj_mat)
            conc = int(model.concentrated)

            df.at[len(df.index)] = [kappa, edges, nodes, dmin, dmax, diam, conc]
    
    return df

num_samples = 10000
nodes = 50
neighbors = 4
probability = 0.80
half_sync = False

df_NWS = sample_NWS(num_samples, nodes, neighbors, probability)
df_NWS