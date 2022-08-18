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


class Data_Gen:
    
    def __init__(self, num_samples = None, NodeNum =None, prob=None, knn=None, kap=None, GHMItNum = None):
        self.num_samples = num_samples
        self.NodeNum = NodeNum
        self.prob = prob
        self.knn = knn
        self.kap = kap
        self.GHMItNum = GHMItNum
    
    def generate_nxg(self, X):
    
        graph_list = []
        k = int(np.sqrt(X.shape[0]))
        for i in range(X.shape[1]):
            adj_mat = X.T[i].reshape(k,k)
            G = nx.from_numpy_matrix(adj_mat)
            graph_list.append(G)
        
        return graph_list
    
    def Equal_Sample_WSGHM(self):
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
        
        SyncNum = 0
        NonSync = 0
        for i in range(self.num_samples):
            print(i)
            G = nx.watts_strogatz_graph(self.NodeNum, self.knn, self.prob)
            s = np.random.randint(5, size=1*self.NodeNum)
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
                GHM_Class = GHM(G=G, S=s, Kap=self.kap, ItNum=self.GHMItNum)
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
                SList.extend(list(s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                if max(SyncNum,NonSync) <= 1225:
                    df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
                    
                elif min(SyncNum,NonSync) <= 1225:
                    if min(SyncNum,NonSync) == SyncNum:
                        if label == 1:
                            df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
                    elif min(SyncNum,NonSync) == NonSync:
                        if label == 0:
                            df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
        
    
        return df, SyncNum


    def Equal_Sample_UCLAGHM(self):
        sampling_alg = 'pivot'

        ntwk = 'UCLA26' # COVID_PPI, Wisconsin87, Caltech36
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
        for G in graph_list:
            print(i)
            s = np.random.randint(5, size=1*k)
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
                GHM_Class = GHM(G=G, S=s, Kap=self.kap, ItNum=self.GHMItNum)
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
                SList.extend(list(s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        
    
        return df, SyncNum
    
    def Equal_Sample_CaltechGHM(self):
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
        for G in graph_list:
            print(i)
            s = np.random.randint(5, size=1*k)
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
                GHM_Class = GHM(G=G, S=s, Kap=self.kap, ItNum=self.GHMItNum)
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
                SList.extend(list(s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        
    
        return df, SyncNum
    
    
    def Equal_Sample_HarvardGHM(self):
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
        for G in graph_list:
            print(i)
            s = np.random.randint(5, size=1*k)
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
                GHM_Class = GHM(G=G, S=s, Kap=self.kap, ItNum=self.GHMItNum)
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
                SList.extend(list(s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        
    
        return df, SyncNum
    
    def Equal_Sample_WisconsinGHM(self):
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
        for G in graph_list:
            print(i)
            s = np.random.randint(5, size=1*k)
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
                GHM_Class = GHM(G=G, S=s, Kap=self.kap, ItNum=self.GHMItNum)
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
                SList.extend(list(s))
                SList.extend(All_Later_States_flatten)
                SList.append(label)
                df.at[len(df.index)] = [self.kap, edges, nodes, dmin, dmax, diam]+SList
            i+=1
        
    
        return df, SyncNum

