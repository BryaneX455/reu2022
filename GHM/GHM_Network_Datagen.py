# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 00:29:20 2022

@author: Bryan
"""
import seaborn as sb
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys; sys.path.insert(1,'../')
import sys; sys.path.insert(1, 'E:/college/year4.1/REU/reu2022-main/ML_for_FCA/') 
from math import floor
from tqdm import tqdm
from firefly import *


def animate_GHM2D(G,S,Kap,ItNum):
    """Implements 2D Greenberg Hasting Model Animation

    Args:
        G (NetworkX Graph): Input graph to the model
        S (array): Current state
        Kap (int): Kap-color GHM
        ItNum (int): Number of iterations

    Returns:
        2D heatmap animation for the GHM
    """
    
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
            NeighbNum2D = len(list(G.neighbors(list(G.nodes)[j])))
            NeighbSet2D = G.neighbors(list(G.nodes)[j])
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
                SN[int((j-j%n)/n)][int(j%n)] = (SN[int((j-j%n)/n)][int(j%n)] + 1) % Kap
    
    
        print(S)

        for j in range(m):
            for k in range(n):                
                PhaseStateList[i][j][k] = S[j][k]
                
        Sync = False
        if not np.any(PhaseStateList[ItNum-1]):
            Sync = True
                           
    return PhaseStateList, 

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
        
        

NV = 6
print('Building...')
gs = make_graphs(NV)
print('Filtering...')
gs = filter(gs, NV)
print(f'Drawing {len(gs)} graphs...')
plot_graphs(gs, figsize=14, dotsize=20)



file = open('toy.csv', 'w+', newline ='')

GHM_iter = 30
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
            quartile_1 = s.quantiles(col, n=4)[0]
            quartile_2 = s.quantiles(col, n=4)[1]
            quartile_3 = s.quantiles(col, n=4)[2]

            sample = [num_edges, num_nodes, min_degree, max_degree, diameter, 
                       quartile_1, quartile_2, quartile_3]
            states, label = FCA(G, col, kappa, FCA_iter)
            sample.append(label)
            for j in range(5):
                sample = sample+list(states[j])
            
            width = width_compute(states[FCA_iter-1],kappa)
            concentration = False
            if(width < floor(kappa/2)): #half circle concentration
                concentration = True
            sample.append(concentration)
            write.writerow(sample)