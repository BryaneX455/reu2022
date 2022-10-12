# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:58:14 2022
@author: Zihong Xu
"""
import math
import numpy as np
import pandas   as pd
import networkx as nx
from GHM_Classes.GHM import GHM
from NNetwork_master.src.NNetwork import NNetwork as nn
from itertools import chain
from statistics import stdev


class Data_Gen:
    
    def __init__(self, num_samples = None, NodeNum =None, prob=None, knn=None, kap=None, GHMItNum = None, s = None):
        self.num_samples = num_samples
        self.NodeNum = NodeNum
        self.prob = prob
        self.knn = knn
        self.kap = kap
        self.GHMItNum = GHMItNum
        self.s = s
    def generate_nxg(self, X):
    
        graph_list = []
        k = int(np.sqrt(X.shape[0]))
        for i in range(X.shape[1]):
            adj_mat = X.T[i].reshape(k,k)
            G = nx.from_numpy_matrix(adj_mat)
            graph_list.append(G)
        
        return graph_list
    
    def Sample_NWSGHM(self):
        BaseName = "S" 
        Edge_Base_Name = "E"
        InitPhaseList = list(np.zeros(self.NodeNum))
        AllPhase = list(np.zeros(int(self.NodeNum*(self.GHMItNum-1))))
        EdgeList = list(np.zeros(int(self.NodeNum*self.NodeNum)))
        for i in range(int(self.NodeNum*self.NodeNum)):
            Row_Pos = math.floor(i/self.NodeNum)
            Col_Pos = i%self.NodeNum
            Row_Pos_Name = str(Row_Pos)
            Col_Pos_Name = str(Col_Pos)
            EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'
    
        for i in range(self.NodeNum):
            InitPhaseList[i] = f'{BaseName}_{i}_{0}'
            
        for i in range(int(self.NodeNum*(self.GHMItNum-1))):
            ItNum = math.floor(i/self.NodeNum)+1
            Col_Pos = i%self.NodeNum
            BaseName = "S"       
            AllPhase[i] = f'{BaseName}_{Col_Pos}_{ItNum}'
            
        ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
        ColList.extend(EdgeList)
        ColList.extend(InitPhaseList)
        ColList.extend(AllPhase)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        G = nx.newman_watts_strogatz_graph(20000, 1000, 0.7)
        edgelist = []
        for i in list(G.edges(data=True)):
            edgelist.append([i[0], i[1]])
        G = nn.NNetwork()
        G.add_edges(edgelist)     
        X, embs = G.get_patches(k=self.NodeNum, sample_size=10000, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        SyncNum = 0
        NonSync = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        for G in graph_list:
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            if nx.is_connected(G):
                # Number of Edges and Nodes
                # EdgeList = G.edges
                edges = G.number_of_edges()
                nodes = G.number_of_nodes()
    
                # Min and Max degree
                degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                dmax = max(degree_sequence)
                dmin = min(degree_sequence)
    
                # Diameter of graph
                diam = nx.diameter(G)
                
                # Applying GHM
                Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
                Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
                GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
                state, label = GHM_Class.GHM1D()
                All_Later_States_flatten = list(chain.from_iterable(state[1:]))
                if label:
                    label = 1
                    SyncNum += 1
                else:
                    label = 0
                    NonSync += 1
                
                SList = []
                SList.extend(Vec_Adj_Matrix)
                SList.extend(list(self.s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
    
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam


    def Sample_UCLAGHM(self):
        sampling_alg = 'pivot'

        ntwk = 'UCLA26' # COVID_PPI, Wisconsin87, Caltech36
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        k=self.NodeNum
        
        path = str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        #print('num nodes in G', len(G.nodes()))
        #print('num edges in G', len(G.get_edges()))
        
        X, embs = G.get_patches(k=self.NodeNum, sample_size=self.num_samples, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        SyncNum = 0
        NonSync = 0
        BaseName = "S" 
        Edge_Base_Name = "E"
        InitPhaseList = list(np.zeros(k))
        AllPhase = list(np.zeros(int(k*(self.GHMItNum-1))))
        EdgeList = list(np.zeros(int(k*k)))
        
        for i in range(int(k*k)):
            Row_Pos = math.floor(i/k)
            Col_Pos = i%k
            Row_Pos_Name = str(Row_Pos)
            Col_Pos_Name = str(Col_Pos)
            EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'
    
        for i in range(k):
            InitPhaseList[i] = f'{BaseName}_{i}_{0}'
            
        for i in range(int(k*(self.GHMItNum-1))):
            ItNum = math.floor(i/k)+1
            Col_Pos = i%k
            BaseName = "S"       
            AllPhase[i] = f'{BaseName}_{Col_Pos}_{ItNum}'
            
        ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
        ColList.extend(EdgeList)
        ColList.extend(InitPhaseList)
        ColList.extend(AllPhase)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        i = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        for G in graph_list:
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            if nx.is_connected(G):
                # Number of Edges and Nodes
                # EdgeList = G.edges
                edges = G.number_of_edges()
                nodes = G.number_of_nodes()
    
                # Min and Max degree
                degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                dmax = max(degree_sequence)
                dmin = min(degree_sequence)
    
                # Diameter of graph
                diam = nx.diameter(G)
                
                # Applying GHM
                Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
                Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
                GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
                state, label = GHM_Class.GHM1D()
                All_Later_States_flatten = list(chain.from_iterable(state[1:]))
                if label:
                    label = 1
                    SyncNum += 1
                else:
                    label = 0
                    NonSync += 1
                
                SList = []
                SList.extend(Vec_Adj_Matrix)
                SList.extend(list(self.s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam
    
    def Sample_CaltechGHM(self):
        sampling_alg = 'pivot'

        ntwk = 'Caltech36' # COVID_PPI, Wisconsin87, Caltech36
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        k=self.NodeNum
        
        path = str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        #print('num nodes in G', len(G.nodes()))
        #print('num edges in G', len(G.get_edges()))
        
        X, embs = G.get_patches(k=k, sample_size=self.num_samples, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        SyncNum = 0
        NonSync = 0
        BaseName = "S" 
        Edge_Base_Name = "E"
        InitPhaseList = list(np.zeros(k))
        AllPhase = list(np.zeros(int(k*(self.GHMItNum-1))))
        EdgeList = list(np.zeros(int(k*k)))
        
        for i in range(int(k*k)):
            Row_Pos = math.floor(i/k)
            Col_Pos = i%k
            Row_Pos_Name = str(Row_Pos)
            Col_Pos_Name = str(Col_Pos)
            EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'
    
        for i in range(k):
            InitPhaseList[i] = f'{BaseName}_{i}_{0}'
            
        for i in range(int(k*(self.GHMItNum-1))):
            ItNum = math.floor(i/k)+1
            Col_Pos = i%k
            BaseName = "S"       
            AllPhase[i] = f'{BaseName}_{Col_Pos}_{ItNum}'
            
        ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
        ColList.extend(EdgeList)
        ColList.extend(InitPhaseList)
        ColList.extend(AllPhase)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        i = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        for G in graph_list:
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            if nx.is_connected(G):
                # Number of Edges and Nodes
                # EdgeList = G.edges
                edges = G.number_of_edges()
                nodes = G.number_of_nodes()
    
                # Min and Max degree
                degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                dmax = max(degree_sequence)
                dmin = min(degree_sequence)
    
                # Diameter of graph
                diam = nx.diameter(G)
                
                # Applying GHM
                Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
                Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
                GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
                state, label = GHM_Class.GHM1D()
                All_Later_States_flatten = list(chain.from_iterable(state[1:]))
                if label:
                    label = 1
                    SyncNum += 1
                else:
                    label = 0
                    NonSync += 1
                
                SList = []
                SList.extend(Vec_Adj_Matrix)
                SList.extend(list(self.s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam
    
    
    def Sample_HarvardGHM(self):
        sampling_alg = 'pivot'

        ntwk = 'Harvard1' # COVID_PPI, Wisconsin87, Caltech36
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        k=self.NodeNum
        
        path = str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        #print('num nodes in G', len(G.nodes()))
        #print('num edges in G', len(G.get_edges()))
        
        X, embs = G.get_patches(k=k, sample_size=self.num_samples, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        SyncNum = 0
        NonSync = 0
        BaseName = "S" 
        Edge_Base_Name = "E"
        InitPhaseList = list(np.zeros(k))
        AllPhase = list(np.zeros(int(k*(self.GHMItNum-1))))
        EdgeList = list(np.zeros(int(k*k)))
        
        for i in range(int(k*k)):
            Row_Pos = math.floor(i/k)
            Col_Pos = i%k
            Row_Pos_Name = str(Row_Pos)
            Col_Pos_Name = str(Col_Pos)
            EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'
    
        for i in range(k):
            InitPhaseList[i] = f'{BaseName}_{i}_{0}'
            
        for i in range(int(k*(self.GHMItNum-1))):
            ItNum = math.floor(i/k)+1
            Col_Pos = i%k
            BaseName = "S"       
            AllPhase[i] = f'{BaseName}_{Col_Pos}_{ItNum}'
            
        ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
        ColList.extend(EdgeList)
        ColList.extend(InitPhaseList)
        ColList.extend(AllPhase)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        i = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        for G in graph_list:
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            if nx.is_connected(G):
                # Number of Edges and Nodes
                # EdgeList = G.edges
                edges = G.number_of_edges()
                nodes = G.number_of_nodes()
    
                # Min and Max degree
                degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                dmax = max(degree_sequence)
                dmin = min(degree_sequence)
    
                # Diameter of graph
                diam = nx.diameter(G)
                
                # Applying GHM
                Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
                Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
                GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
                state, label = GHM_Class.GHM1D()
                All_Later_States_flatten = list(chain.from_iterable(state[1:]))
                if label:
                    label = 1
                    SyncNum += 1
                else:
                    label = 0
                    NonSync += 1
                
                SList = []
                SList.extend(Vec_Adj_Matrix)
                SList.extend(list(self.s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam
    
    def Sample_WisconsinGHM(self):
        sampling_alg = 'pivot'

        ntwk = 'Wisconsin87' # COVID_PPI, Wisconsin87, Caltech36
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        k=self.NodeNum
        
        path = str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        #print('num nodes in G', len(G.nodes()))
        #print('num edges in G', len(G.get_edges()))
        
        X, embs = G.get_patches(k=k, sample_size=self.num_samples, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        SyncNum = 0
        NonSync = 0
        BaseName = "S" 
        Edge_Base_Name = "E"
        InitPhaseList = list(np.zeros(k))
        AllPhase = list(np.zeros(int(k*(self.GHMItNum-1))))
        EdgeList = list(np.zeros(int(k*k)))
        
        for i in range(int(k*k)):
            Row_Pos = math.floor(i/k)
            Col_Pos = i%k
            Row_Pos_Name = str(Row_Pos)
            Col_Pos_Name = str(Col_Pos)
            EdgeList[i] = f'{Edge_Base_Name}_{Row_Pos_Name}_{Col_Pos_Name}'
    
        for i in range(k):
            InitPhaseList[i] = f'{BaseName}_{i}_{0}'
            
        for i in range(int(k*(self.GHMItNum-1))):
            ItNum = math.floor(i/k)+1
            Col_Pos = i%k
            BaseName = "S"       
            AllPhase[i] = f'{BaseName}_{Col_Pos}_{ItNum}'
            
        ColList = ['kappa', '# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter']
        ColList.extend(EdgeList)
        ColList.extend(InitPhaseList)
        ColList.extend(AllPhase)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        i = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        for G in graph_list:
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            if nx.is_connected(G):
                # Number of Edges and Nodes
                # EdgeList = G.edges
                edges = G.number_of_edges()
                nodes = G.number_of_nodes()
    
                # Min and Max degree
                degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                dmax = max(degree_sequence)
                dmin = min(degree_sequence)
    
                # Diameter of graph
                diam = nx.diameter(G)
                
                # Applying GHM
                Adj_Matrix = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
                Vec_Adj_Matrix = list(np.asarray(Adj_Matrix).astype(int).flatten())
                GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
                state, label = GHM_Class.GHM1D()
                All_Later_States_flatten = list(chain.from_iterable(state[1:]))
                if label:
                    label = 1
                    SyncNum += 1
                else:
                    label = 0
                    NonSync += 1
                
                SList = []
                SList.extend(Vec_Adj_Matrix)
                SList.extend(list(self.s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam


    def Sample_3D(self, Network):
        sampling_alg = 'pivot'
    
        ntwk = Network # COVID_PPI, Wisconsin87, Caltech36
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        k=self.NodeNum
        
        path = str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        
        X, embs = G.get_patches(k=k, sample_size=self.num_samples, skip_folded_hom=True)
        graph_list = self.generate_nxg(X)
        
        Dyn_Num = 5
        BaseName = "D" 
        Dyn_Diff_List = list(np.zeros(int(k*k)*Dyn_Num))
        
        for i in range(Dyn_Num):     
            for j in range(int(k*k)):
                Row_Pos = int(j//k)
                Col_Pos = int(j%k)
                Node1 = str(Col_Pos + 1)
                Node2 = str(Row_Pos + 1) 
                Dyn_Diff_List[i] = f'{BaseName}_{Node1}_{Node2}_{i}'
                
            
            
        ColList = []
        ColList.extend(Dyn_Diff_List)
        ColList.extend(['label'])
        df = pd.DataFrame(columns=ColList)
        i = 0
        Total_Edge = 0
        Edge_Num_List = []
        Total_Diam = 0
        Diameter_List = []
        SyncNum = 0
        NonSync = 0
        for G in graph_list:
            GHM_Class = GHM(G=G, S=self.s, Kap=self.kap, ItNum=self.GHMItNum)
            Edge_Num_List.append(G.number_of_edges())
            Total_Edge += G.number_of_edges()
            Total_Diam += nx.diameter(G)
            Diameter_List.append(nx.diameter(G))
            Full_Mat, Sync = GHM_Class.GHM_Adj_Color_Diff( G, k, self.kap, self.GHMItNum, ntwk)
            Full_Mat = np.array(Full_Mat)
            if Sync:
                label = 1
            else:
                label = 0
            # Flatten out Full Matrix
            FM_Flatten = (Full_Mat.flatten().reshape(Full_Mat.shape[0],Full_Mat.shape[1]*Full_Mat.shape[2])).flatten()
            # Extend list in dataframe
            SList = []
            SList.extend(FM_Flatten)
            SList.append(label)
            print(len(df.index))
            df.at[len(df.index)] = SList
            
            
        Ave_Edge = Total_Edge/self.num_samples
        Std_Edge = stdev(Edge_Num_List)
        Ave_Diam = Total_Diam/self.num_samples
        Std_Diam = stdev(Diameter_List)
        return df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam
