import pandas as pd
import numpy as np
from scipy. sparse import csr_matrix
import random

import networkx as nx
from kuramoto import Kuramoto

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rf

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from karateclub import Graph2Vec

import warnings
warnings.filterwarnings("ignore")


def gen_all(nodes, r, num_samples, edge_var="L", random_K=False, half_sync=False):
    
    df = pd.DataFrame()
    graph_list = []
    
    headers = []
    for i in range(1, nodes+1):
        for j in range(1, r+1):
            headers.append(f's{i}_{j}')
    
    for i in range(num_samples):
        
        df1 = pd.DataFrame(columns=['# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
        temp = pd.DataFrame()
        
        nodes = nodes
        
        assert(edge_var in ["L", "H", "F"]), "The edge variance can either be \"L\" (Low), \"H\" (High), or \"F\" (Fixed)."
        
        if edge_var=="L":
            neighbors = int(random.uniform(15, 20))
            
        elif edge_var=="H":
            neighbors = int(random.uniform(1, 20))
            
        else:
            neighbors = 15
            
        probability = random.uniform(0, 1)
        G = nx.newman_watts_strogatz_graph(nodes, neighbors, probability)
        graph_list.append(G)
        
        if random_K: K = random.uniform(0.5, 4.5)
        else: K = 1.96
            
        
        if nx.is_connected(G):
            
            # Generate adjacency matrix
            matrix = nx.adjacency_matrix(G)
            matrix = csr_matrix(matrix).toarray().flatten(order='C')
            df_matrix = pd.DataFrame(matrix).T
            
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
                model = Kuramoto(coupling=K, dt=0.01, T=25, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = Kuramoto(coupling=K, dt=0.01, T=25, n_nodes=nodes, half_sync=half_sync)

            sim = model.run(adj_mat)
            conc = int(model.concentrated)
            
            df1.at[len(df1.index)] = [edges, nodes, dmin, dmax, diam, conc]
            
            df2 = pd.DataFrame(sim[:,:r].flatten(order='C')).T
            df2.columns = headers
            
            temp = pd.concat([df_matrix, df2, df1], axis=1)

            df = pd.concat([df, temp], ignore_index=True)
            
        else:
            graph_list.remove(G)
            
    g2v = Graph2Vec(wl_iterations = 5, dimensions = 16)
    g2v.fit(graph_list)
    embed = g2v.get_embedding()
    df_g2v = pd.DataFrame(embed, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 
                                          'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16'])
        
    df_final = pd.concat([df_g2v, df], axis=1)
            
    return df_final
        
        
def gen_KM(nodes, r, num_samples, edge_var="L", random_K=False, half_sync=False):
    
    df = pd.DataFrame()
    
    for i in range(num_samples):
        
        df1 = pd.DataFrame(columns=['# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
        temp = pd.DataFrame()
        
        nodes = nodes

        assert(edge_var in ["L", "H", "F"]), "The edge variance can either be \"L\" (Low), \"H\" (High), or \"F\" (Fixed)."
        
        if edge_var=="L":
            neighbors = int(random.uniform(15, 20))
            
        elif edge_var=="H":
            neighbors = int(random.uniform(1, 20))
            
        else:
            neighbors = 15
            
        probability = random.uniform(0, 1)
        G = nx.newman_watts_strogatz_graph(nodes, neighbors, probability)
        
        if random_K: K = random.uniform(0.5, 4.5)
        else: K = 1.96
        
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
                model = Kuramoto(coupling=K, dt=0.01, T=25, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = Kuramoto(coupling=K, dt=0.01, T=25, n_nodes=nodes, half_sync=half_sync)

            sim = model.run(adj_mat)
            conc = int(model.concentrated)
            
            df1.at[len(df1.index)] = [edges, nodes, dmin, dmax, diam, conc]

            df2 = pd.DataFrame(sim[:,:r].flatten(order='C')).T
            temp = pd.concat([df2, df1], axis=1)

            df = pd.concat([df, temp], ignore_index=True)
    
    return df


def gen_KM_nodynamics(nodes, num_samples, edge_var="L", random_K=False, half_sync=False):
    
    df = pd.DataFrame(columns=['# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
    
    for i in range(num_samples):
        nodes = nodes
        
        assert(edge_var in ["L", "H", "F"]), "The edge variance can either be \"L\" (Low), \"H\" (High), or \"F\" (Fixed)."
        
        if edge_var=="L":
            neighbors = int(random.uniform(15, 20))
            
        elif edge_var=="H":
            neighbors = int(random.uniform(1, 20))
            
        else:
            neighbors = 15
        
        probability = random.uniform(0, 1)
        G = nx.newman_watts_strogatz_graph(nodes, neighbors, probability)
        
        if random_K: K = random.uniform(0.5, 4.5)
        else: K = 1.96
        
        if nx.is_connected(G):
            edges = G.number_of_edges()
            nodes = G.number_of_nodes()
            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)
            diam = nx.diameter(G)

            adj_mat = nx.to_numpy_array(G)

            if half_sync:
                natfreqs = np.repeat(2 * np.pi * 0, nodes)
                model = Kuramoto(coupling=K, dt=0.01, T=25, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = Kuramoto(coupling=K, dt=0.01, T=25, n_nodes=nodes, half_sync=half_sync)

            sim = model.run(adj_mat)
            conc = int(model.concentrated)
            
            df.at[len(df.index)] = [edges, nodes, dmin, dmax, diam, conc]
    
    return df


def g2v_KM(nodes, r, num_samples, edge_var="L", random_K=False, half_sync=False):
    
    graph_list = []
    
    df = pd.DataFrame()
    
    for i in range(num_samples):
        
        df1 = pd.DataFrame(columns=['# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
        temp = pd.DataFrame()
        
        nodes = nodes

        assert(edge_var in ["L", "H", "F"]), "The edge variance can either be \"L\" (Low), \"H\" (High), or \"F\" (Fixed)."
        
        if edge_var=="L":
            neighbors = int(random.uniform(15, 20))
            
        elif edge_var=="H":
            neighbors = int(random.uniform(1, 20))
            
        else:
            neighbors = 15
            
        probability = random.uniform(0, 1)
        G = nx.newman_watts_strogatz_graph(nodes, neighbors, probability)
        graph_list.append(G)
        
        if random_K: K = random.uniform(0.5, 4.5)
        else: K = 1.96
        
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
                model = Kuramoto(coupling=K, dt=0.01, T=25, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = Kuramoto(coupling=K, dt=0.01, T=25, n_nodes=nodes, half_sync=half_sync)

            sim = model.run(adj_mat)
            conc = int(model.concentrated)
            
            df1.at[len(df1.index)] = [edges, nodes, dmin, dmax, diam, conc]

            df2 = pd.DataFrame(sim[:,:r].flatten(order='C')).T
            temp = pd.concat([df2, df1], axis=1)

            df = pd.concat([df, temp], ignore_index=True)
            
        else:
            graph_list.remove(G)
            
    g2v = Graph2Vec(wl_iterations = 5, dimensions = 16)
    g2v.fit(graph_list)
    embed = g2v.get_embedding()
    df_g2v = pd.DataFrame(embed, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 
                                         'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16'])
        
    df_final = pd.concat([df_g2v, df], axis=1)
    
    return df_final


def g2v_KM_nodynamics(nodes, num_samples, edge_var="L", random_K=False, half_sync=False):
    
    graph_list = []
    df = pd.DataFrame(columns=['# Edges', '# Nodes', 'Min Degree', 'Max Degree', 'Diameter', 'Concentrated'])
    
    for i in range(num_samples):
        nodes = nodes
        
        assert(edge_var in ["L", "H", "F"]), "The edge variance can either be \"L\" (Low), \"H\" (High), or \"F\" (Fixed)."
        
        if edge_var=="L":
            neighbors = int(random.uniform(15, 20))
            
        elif edge_var=="H":
            neighbors = int(random.uniform(1, 20))
            
        else:
            neighbors = 15
        
        probability = random.uniform(0, 1)
        G = nx.newman_watts_strogatz_graph(nodes, neighbors, probability)
        graph_list.append(G)
        
        if random_K: K = random.uniform(0.5, 4.5)
        else: K = 1.96
        
        if nx.is_connected(G):
            edges = G.number_of_edges()
            nodes = G.number_of_nodes()
            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)
            diam = nx.diameter(G)

            adj_mat = nx.to_numpy_array(G)

            if half_sync:
                natfreqs = np.repeat(2 * np.pi * 0, nodes)
                model = Kuramoto(coupling=K, dt=0.01, T=25, natfreqs=natfreqs, half_sync=half_sync)
            else:
                model = Kuramoto(coupling=K, dt=0.01, T=25, n_nodes=nodes, half_sync=half_sync)

            sim = model.run(adj_mat)
            conc = int(model.concentrated)
            
            df.at[len(df.index)] = [edges, nodes, dmin, dmax, diam, conc]
            
        else:
            graph_list.remove(G)
            
    g2v = Graph2Vec(wl_iterations = 5, dimensions = 16)
    g2v.fit(graph_list)
    embed = g2v.get_embedding()
    df_g2v = pd.DataFrame(embed, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 
                                          'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16'])
        
    df_final = pd.concat([df_g2v, df], axis=1)
    
    return df_final

def generate_data(df, oversample=True):
    X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    y = y.astype('int')

    if oversample:
        sampler = RandomOverSampler()
    else:
        sampler = RandomUnderSampler()
        
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                        test_size = 0.25,
                                                        stratify = y_resampled)
    
    return X_train, X_test, y_train, y_test

def gen_g2v_nodynamics_data(df, oversample = True):
    X, y = df.iloc[:, 0:16], df.iloc[:, [-1]]
    y = y.astype('int')
    
    if oversample:
        sampler = RandomOverSampler()
    else:
        sampler = RandomUnderSampler()
        
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                        test_size = 0.25,
                                                        stratify = y_resampled)
    
    return X_train, X_test, y_train, y_test

def gen_g2v_data(df, oversample = True):
    X, y = df.iloc[:, :-6], df.iloc[:, [-1]]
    y = y.astype('int')
    
    if oversample:
        sampler = RandomOverSampler()
    else:
        sampler = RandomUnderSampler()
        
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                        test_size = 0.25,
                                                        stratify = y_resampled)
    
    return X_train, X_test, y_train, y_test

def model_metrics(model, y_test, y_pred):
    conf_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()
    
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('F1: %.3f' % f1_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    
    pass
    return
    
    
def plot_gini_index(model, X_train):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0).reshape((-1,1))
    feature_names = list(X_train.columns)

    forest_importances = pd.Series(importances, index=feature_names)
    if len(forest_importances) > 100:
        forest_importances = forest_importances.sort_values(ascending=False).head(50)

    fig, ax = plt.subplots(figsize=(10,8))
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using Gini index")
    ax.set_ylabel("Mean decrease in impurity (Gini index)")
    fig.tight_layout()