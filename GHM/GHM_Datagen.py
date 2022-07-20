# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:31:21 2022

@author: Bryan
"""

import csv
import numpy as np
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sb
import statistics as st

from NNetwork.NNetwork import NNetwork as NNT
from itertools import product

def make_graphs(n=2, i=None, j=None):
    """Make a graph recursively, by either including, or skipping each edge.

    Edges are given in lexicographical order by construction."""
    out = []
    if i is None: # First call

        out  = [[(0,1)]+r for r in make_graphs(n=n, i=0, j=1)]
    elif j<n-1:
        out += [[(i,j+1)]+r for r in make_graphs(n=n, i=i, j=j+1)]
        out += [          r for r in make_graphs(n=n, i=i, j=j+1)]
    elif i<n-1:
        out = make_graphs(n=n, i=i+1, j=i+1)
    else:
        out = [[]]
    return out

def perm(n, s=None):
    """All permutations of n elements."""
    if s is None: return perm(n, tuple(range(n)))
    if not s: return [[]]
    return [[i]+p for i in s for p in perm(n, tuple([k for k in s if k!=i]))]

def permute(g, n):
    """Create a set of all possible isomorphic codes for a graph,

    as nice hashable tuples. All edges are i<j, and sorted lexicographically."""
    ps = perm(n)
    out = set([])
    for p in ps:
        out.add(tuple(sorted([(p[i],p[j]) if p[i]<p[j]
                              else (p[j],p[i]) for i,j in g])))
    return list(out)

def connected(g):
    """Check if the graph is fully connected, with Union-Find."""
    nodes = set([i for e in g for i in e])
    roots = {node: node for node in nodes}

    def _root(node, depth=0):
        if node==roots[node]: return (node, depth)
        else: return _root(roots[node], depth+1)

    for i,j in g:
        ri,di = _root(i)
        rj,dj = _root(j)
        if ri==rj: continue
        if di<=dj: roots[ri] = rj
        else:      roots[rj] = ri
    return len(set([_root(node)[0] for node in nodes]))==1

def filter(gs, target_nv):
    """Filter all improper graphs: those with not enough nodes,

    those not fully connected, and those isomorphic to previously considered."""
    mem = set({})
    gs2 = []
    for g in gs:
        nv = len(set([i for e in g for i in e]))
        if nv != target_nv:
            continue
        if not connected(g):
            continue
        if tuple(g) not in mem:
            gs2.append(g)
            mem |= set(permute(g, target_nv))
    return gs2

# Main body


def plot_graphs(graphs, figsize=14, dotsize=20):
    """Utility to plot a lot of graphs from an array of graphs.

    Each graphs is a list of edges; each edge is a tuple."""
    n = len(graphs)
    fig = plt.figure(figsize=(figsize,figsize))
    fig.patch.set_facecolor('white') # To make copying possible (white background)

    k = int(np.sqrt(n))
    for i in range(n):
        plt.subplot(k+1,k+1,i+1)
        g = nx.Graph() # Generate a Networkx object

        for e in graphs[i]:            
            g.add_edge(e[0],e[1])
        nx.draw_kamada_kawai(g, node_size=dotsize)
        print('.', end='')


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

NodeNum = 6
print('Building...')
gs = make_graphs(NodeNum)
print('Filtering...')
gs = filter(gs, NodeNum)
#print(f'Drawing {len(gs)} graphs...')
#plot_graphs(gs, figsize=14, dotsize=20)
PermNum = len(gs)
#print(PermNum)
#print(gs.type)
    
kap = 5
#s = np.random.randint(kap, size=NodeNum)
color_list = list(product(range(0, kap), repeat=NodeNum))
print(len(color_list))
file = open('GHM.csv', 'w+', newline ='')
GHMItNum = 30
with file:
    write = csv.writer(file)
    for col in color_list:
        for i in gs:
            G = nx.Graph()
            G.add_edges_from(i)

            num_edges = G.number_of_edges()
            #num_nodes = G.number_of_nodes()
            min_degree = min(list(G.degree), key=lambda x:x[1])[1]
            max_degree = max(list(G.degree), key=lambda x:x[1])[1]
            diameter = nx.diameter(G)
            quartile_1 = st.quantiles(col, n=4)[0]
            quartile_2 = st.quantiles(col, n=4)[1]
            quartile_3 = st.quantiles(col, n=4)[2]

            sample = [num_edges, NodeNum, min_degree, max_degree, diameter, 
                       quartile_1, quartile_2, quartile_3]
            states, label = GHM(G, col, kap, GHMItNum)
            sample.append(label)
            for j in range(5):
                sample = sample+list(states[j])
            
            #width = width_compute(states[FCA_iter-1],kappa)
            #concentration = False
            #if(width < floor(kappa/2)): #half circle concentration
            #    concentration = True
            #sample.append(concentration)
            write.writerow(sample)
    