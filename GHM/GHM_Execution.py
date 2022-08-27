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
# from GHM_Classes.GHM_Animation import GHM_Animation
from GHM_Classes.GHM_WS_Datagen_AllStates import Data_Gen
from NNetwork_master.src.NNetwork import NNetwork as nn

Data_Gen_Class = Data_Gen()
NodeNum = 20
knn = 4
prob= 0.65
Kap = 5
ItNum = 30


"""Watts Strogatz 1-D"""
G = nx.watts_strogatz_graph(NodeNum, knn, prob)
s = np.random.randint(5, size=1*NodeNum)
GHM_Class = GHM(G, s, Kap, ItNum)
st, sync = GHM_Class.GHM1D()
sb.heatmap(st)
plt.title('GHM1D_WS_20Nodes_4knn_0.65Rewiring')
plt.show()

"""Networkx 2-D grid"""
G = nx.grid_2d_graph(8, 8) 
s = np.random.randint(5, size=1*64)
GHM_Class = GHM(G, s, Kap, ItNum)
PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr()
sb.heatmap(PhaseStateDf, cbar=False, cmap='viridis')
plt.title('GHM2D_64Nodes_Sample')
plt.show()

"""Networkx 2-D grid Animastion"""
"""
G = nx.grid_2d_graph(70, 70) 
s = np.random.randint(5, size=(70,70))
GHM_Animation_Class = GHM_Animation(G, s, Kap, ItNum)
GHM_Animation_Class.Animate_GHM2D()
"""


"""Graph with different number of nodes"""
sampling_alg = 'pivot'
#ncol = 4
#nrow = 2
#fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))
ntwk ='UCLA26' # COVID_PPI, Wisconsin87, Caltech36
ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
Node_Num_Min = 5
Node_Num_Max = 25
num_samples = 10
NodeNum_List = []
Average_Sync_List = []
Ave_Cycle_List = []
for i in range(Node_Num_Min, Node_Num_Max):
    NodeNum_List.append(Node_Num_Min+i)


for i in range(len(NodeNum_List)):
    print(i)
    k = NodeNum_List[i]   
    path = str(ntwk) + '.txt'
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)        
    X, embs = G.get_patches(k=k, sample_size=num_samples, skip_folded_hom=True)
    graph_list = Data_Gen_Class.generate_nxg(X)
    Total_Sync = 0
    Total_Cycle = 0
    for G in graph_list:
        G_Di = G.to_directed()
        Cycle_List = list(nx.simple_cycles(G_Di))
        Cycle_Num = len(Cycle_List)
        Total_Cycle += Cycle_Num
        s = np.random.randint(5, size=1*k)
        GHM_Class = GHM(G=G, S=s, Kap=Kap, ItNum=ItNum)
        state, label = GHM_Class.GHM1D()
        if label:
            Total_Sync += 1
    Ave_Cycle = Total_Cycle/num_samples
    Ave_Cycle_List.append(Ave_Cycle)
    Average_Sync_Perc =  Total_Sync/num_samples
    Average_Sync_List.append(Average_Sync_Perc)
plt.plot(Average_Sync_List)
plt.xlabel('Node Number')
plt.ylabel('Sync Ratio')
plt.show()
plt.plot(Ave_Cycle_List)
plt.xlabel('Node Number')
plt.ylabel('Average Cycle Number')




""" GHM 2D implementation with different node number with assignment probability 1"""
SideLength_List = []
Side_Length_Min = 2
Side_Length_Max = 40
for i in range(Side_Length_Min, Side_Length_Max):
    SideLength_List.append(Side_Length_Min+i)
SyncNum_List = []

for i in range(len(SideLength_List)):   
    print(i)
    G = nx.grid_2d_graph(SideLength_List[i], SideLength_List[i]) 
    s = np.random.randint(5, size=1*SideLength_List[i]**2)
    GHM_Class = GHM(G, s, Kap, ItNum)
    PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr_Stochastic(1)
    if Sync:
        SyncNum_List.append(1)
    else:
        SyncNum_List.append(0)
        
plt.plot(SyncNum_List) 
plt.xlabel('Grid Graph Side length')
plt.xlabel('Sync Ratio')
plt.title('GHM 2D stochastic with assignment probability 1')  
plt.ylim(0,1)
plt.show()     

""" GHM 2D implementation with different node number with assignment probability 0.8"""
SideLength_List = []
Side_Length_Min = 2
Side_Length_Max = 40
for i in range(Side_Length_Min, Side_Length_Max):
    SideLength_List.append(Side_Length_Min+i)
SyncNum_List = []

for i in range(len(SideLength_List)):    
    G = nx.grid_2d_graph(SideLength_List[i], SideLength_List[i]) 
    s = np.random.randint(5, size=1*SideLength_List[i]**2)
    GHM_Class = GHM(G, s, Kap, ItNum)
    PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr_Stochastic(0.8)
    if Sync:
        SyncNum_List.append(1)
    else:
        SyncNum_List.append(0)
        
plt.plot(SyncNum_List) 
plt.xlabel('Grid Graph Side length')
plt.xlabel('Sync Ratio')
plt.title('GHM 2D stochastic with assignment probability 0.8')  
plt.ylim(0,1)
plt.show()


""" GHM 2D implementation with different node number with assignment probability 0.6"""
SideLength_List = []
Side_Length_Min = 2
Side_Length_Max = 40
for i in range(Side_Length_Min, Side_Length_Max):
    SideLength_List.append(Side_Length_Min+i)
SyncNum_List = []

for i in range(len(SideLength_List)):    
    G = nx.grid_2d_graph(SideLength_List[i], SideLength_List[i]) 
    s = np.random.randint(5, size=1*SideLength_List[i]**2)
    GHM_Class = GHM(G, s, Kap, ItNum)
    PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr_Stochastic(0.6)
    if Sync:
        SyncNum_List.append(1)
    else:
        SyncNum_List.append(0)
        
plt.plot(SyncNum_List) 
plt.xlabel('Grid Graph Side length')
plt.xlabel('Sync Ratio')
plt.title('GHM 2D stochastic with assignment probability 0.6')  
plt.ylim(0,1)
plt.show()

""" GHM 2D implementation with different node number with assignment probability 0.4"""
SideLength_List = []
Side_Length_Min = 2
Side_Length_Max = 40
for i in range(Side_Length_Min, Side_Length_Max):
    SideLength_List.append(Side_Length_Min+i)
SyncNum_List = []

for i in range(len(SideLength_List)):    
    G = nx.grid_2d_graph(SideLength_List[i], SideLength_List[i]) 
    s = np.random.randint(5, size=1*SideLength_List[i]**2)
    GHM_Class = GHM(G, s, Kap, ItNum)
    PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr_Stochastic(0.4)
    if Sync:
        SyncNum_List.append(1)
    else:
        SyncNum_List.append(0)
        
plt.plot(SyncNum_List) 
plt.xlabel('Grid Graph Side length')
plt.xlabel('Sync Ratio')
plt.title('GHM 2D stochastic with assignment probability 0.4')  
plt.ylim(0,1)
plt.show()  

""" GHM 2D implementation with different node number with assignment probability 0.2"""
SideLength_List = []
Side_Length_Min = 2
Side_Length_Max = 40
for i in range(Side_Length_Min, Side_Length_Max):
    SideLength_List.append(Side_Length_Min+i)
SyncNum_List = []

for i in range(len(SideLength_List)):    
    G = nx.grid_2d_graph(SideLength_List[i], SideLength_List[i]) 
    s = np.random.randint(5, size=1*SideLength_List[i]**2)
    GHM_Class = GHM(G, s, Kap, ItNum)
    PhaseStateDf, St, Sync = GHM_Class.GHMGridToArr_Stochastic(0.2)
    if Sync:
        SyncNum_List.append(1)
    else:
        SyncNum_List.append(0)
        
plt.plot(SyncNum_List) 
plt.xlabel('Grid Graph Side length')
plt.xlabel('Sync Ratio')
plt.title('GHM 2D stochastic with assignment probability 0.2')  
plt.ylim(0,1)
plt.show()
