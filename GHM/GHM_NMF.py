# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:36:07 2022

@author: Bryan
"""

#Imports
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from NNetwork import NNetwork as nn
from sklearn import svm
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
from sklearn.datasets import make_sparse_coded_signal
from SDL_src.SNMF import SNMF
warnings.filterwarnings("ignore")


def display_graphs(title,
                     save_path,
                     grid_shape=[2,3],
                     fig_size=[10,10],
                     data = None, # [X, embs]
                     show_importance=False):

        # columns of X = vectorized k x k adjacency matrices
        # corresponding list in embs = sequence of nodes (may overalp)
        X, embs = data
        print('X.shape', X.shape)

        rows = grid_shape[0]
        cols = grid_shape[1]

        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        # make outer gridspec

        idx = np.arange(X.shape[1])
        outer_grid = gridspec.GridSpec(nrows=rows, ncols=cols, wspace=0.02, hspace=0.02)

        # make nested gridspecs
        for i in range(rows * cols):
            a = i // cols
            b = i % rows

            Ndict_wspace = 0.05
            Ndict_hspace = 0.05

            # display graphs
            inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)

            # get rid of duplicate nodes
            A = X[:,idx[i]]
            A = X[:,idx[i]].reshape(int(np.sqrt(X.shape[0])), -1)
            H = nn.NNetwork()
            H.read_adj(A, embs[idx[i]])
            A_sub = H.get_adjacency_matrix()

            # read in as a nx graph for plotting
            G1 = nx.from_numpy_matrix(A_sub)
            ax = fig.add_subplot(inner_grid[0, 0])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            weights = [1*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=20, ax=ax, width=weights, label='Graph')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')

sampling_alg = 'pivot'
ntwk = 'Caltech36' # COVID_PPI, Wisconsin87, UCLA26
ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
save_folder = 'GHM_ML'
k=20

path = str(ntwk) + '.txt'
G = nn.NNetwork()
G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
print('num nodes in G', len(G.nodes()))
print('num edges in G', len(G.get_edges()))

# mx0 = G.get_adjacency_matrix(ordered_node_list=G.nodes())
# plt.imshow(mx0)

X, embs = G.get_patches(k=k, sample_size=10000, skip_folded_hom=True)

display_graphs(title='induced subgraphs on {}-walks in {}'.format(k, ntwk_nonumber),
                 save_path=save_folder + ntwk_nonumber + "_subgraphs_"+ str(sampling_alg) + "_walk", 
                 data = [X, embs],
                 grid_shape = [5, 15],
                 fig_size = [15, 5],
                 show_importance=False)

def coding(X, W, H0, 
          r=None, 
          a1=0, #L1 regularizer
          a2=0, #L2 regularizer
          sub_iter=[5], 
          stopping_grad_ratio=0.0001, 
          nonnegativity=True,
          subsample_ratio=1):
    """
    Find \hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) within radius r from H0
    Use row-wise projected gradient descent
    """
    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[1])
    if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
        idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
    A = W.T @ W ## Needed for gradient computation
    grad = W.T @ (W @ H0 - X)
    while (i < np.random.choice(sub_iter)):
        step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
        H1 -= step_size * grad 
        if nonnegativity:
            H1 = np.maximum(H1, 0)  # nonnegativity constraint
        i = i + 1
    return H1

def ALS(X,
        n_components = 10, # number of columns in the dictionary matrix W
        n_iter=100,
        a0 = 0, # L1 regularizer for H
        a1 = 0, # L1 regularizer for W
        a12 = 0, # L2 regularizer for W
        H_nonnegativity=True,
        W_nonnegativity=True,
        compute_recons_error=False,
        subsample_ratio = 10):
    
        '''
        Given data matrix X, use alternating least squares to find factors W,H so that 
                                || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
        is minimized (at least locally)
        '''
        
        d, n = X.shape
        r = n_components
        
        #normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
        #print('!!! avg entry of X', normalization)
        #X = X/normalization

        # Initialize factors 
        W = np.random.rand(d,r)
        H = np.random.rand(r,n) 
        # H = H * np.linalg.norm(X) / np.linalg.norm(H)
        for i in trange(n_iter):
            #H = coding_within_radius(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            #W = coding_within_radius(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            H = coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            W = coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            W /= np.linalg.norm(W)
            if compute_recons_error and (i % 10 == 0) :
                print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
        return W, H


def display_dictionary(W, save_name=None, score=None, grid_shape=None, figsize=[10,10]):
    k = int(np.sqrt(W.shape[0]))
    rows = int(np.sqrt(W.shape[1]))
    cols = int(np.sqrt(W.shape[1]))
    if grid_shape is not None:
        rows = grid_shape[0]
        cols = grid_shape[1]
    
    figsize0=figsize
    if (score is None) and (grid_shape is not None):
        figsize0=(cols, rows)
    if (score is not None) and (grid_shape is not None):
        figsize0=(cols, rows+0.2)
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize0,
                            subplot_kw={'xticks': [], 'yticks': []})
        
    for ax, i in zip(axs.flat, range(W.shape[1])):
        if score is not None:
            idx = np.argsort(score)
            idx = np.flip(idx)    
            
            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="viridis", interpolation='nearest')
            ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)
        else: 
            ax.imshow(W.T[i].reshape(k, k), cmap="viridis", interpolation='nearest')
            if score is not None:
                ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)
       
    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    if save_name is not None:
        plt.savefig( save_name, bbox_inches='tight')
    plt.show()
    

WSD = pd.read_csv('GHM_Dict_Data.csv')
print('WSD:', WSD)
WSD.keys()
b = WSD.copy()
list_keys = list(WSD.keys())

print('list_keys:', list_keys[:25])

y = b['label']
df_True = b[y == 1]
df_False = b[y == 0]
c_True = df_True.loc[:, 'S_0_1':'E_15_15']
c_False = df_False.loc[:,'S_0_1':'E_15_15']
X_True = c_True.values.transpose()
X_False = c_False.values.transpose()

W_True, H_True = ALS(X=X_True, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_True: {X_True.shape}\nShape of W_True: {W_True.shape}\nShape of H_True: {H_True.shape}")

display_dictionary(X_True[16:,:225], figsize=[15,15])

display_dictionary(W_True[16:], figsize=[10,10])

W_False, H_False = ALS(X=X_False, 
           n_components=16, # Reconstruction Error reduces as n_components increases
           n_iter=100, 
           a0 = 0, # L1 regularizer for H
           a1 = 0, # L1 regularizer for W
           a12 = 0, # L2 regularizer for W
           H_nonnegativity=True,
           W_nonnegativity=True,
           compute_recons_error=True,
           subsample_ratio=1)

print(f"Shape of X_False: {X_False.shape}\nShape of W_False: {W_False.shape}\nShape of H_False: {H_False.shape}")

display_dictionary(X_False[16:,:225], figsize=[15,15])

display_dictionary(W_False[16:], figsize=[10,10])


def plot_adj_to_graph_deg(n, weighted, filename=None):
    Subhraph_Col_Num = n
    Subhraph_Row_Num = int(np.ceil(n/2))
    fig, axs = plt.subplots(ncols=Subhraph_Col_Num, nrows=Subhraph_Row_Num, figsize=(Subhraph_Col_Num*5, Subhraph_Row_Num*5))
    Reshape_True_Width = int(np.sqrt(len(W_True[16:])))
    Reshape_False_Width = int(np.sqrt(len(W_False[16:])))
    print(Reshape_True_Width)
    if weighted:
        for i in range(n):
            df_adj = pd.DataFrame(W_True[16:].T[i].reshape(Reshape_True_Width, Reshape_True_Width))
            G = nx.Graph()
            G = nx.from_pandas_adjacency(df_adj)
            edges = G.edges()
            weights = [250*G[u][v]['weight'] for u,v in edges]
            print(weights)
            nx.draw(G, ax=axs[(i//4)*2,i%4], width=weights)
            axs[(i//4)*2,i%4].title.set_text('Synchronizing')
            deg_seq = sorted((d for n, d in G.degree(weight='weight')), reverse=True)
            axs[(i//4)*2+1,i%4].plot(deg_seq, "b-", marker="o")

        for i in range(n):
            df_adj = pd.DataFrame(W_False[16:].T[i].reshape(Reshape_False_Width, Reshape_False_Width))
            G = nx.Graph()
            G = nx.from_pandas_adjacency(df_adj)
            print(G)
            edges = G.edges()
            weights = [250*G[u][v]['weight'] for u,v in edges]
            nx.draw(G, ax=axs[(i//4)*2,i%4+4], width=weights)
            axs[(i//4)*2,i%4+4].title.set_text('Non-Synchronizing')
            deg_seq = sorted((d for n, d in G.degree(weight='weight')), reverse=True)
            axs[(i//4)*2+1,i%4+4].plot(deg_seq, "b-", marker="o")
        if filename != None:
            fig.savefig(filename)
        plt.show()
        
    else:
        for i in range(n):
            df_adj = pd.DataFrame(W_True[16:].T[i].reshape(Reshape_True_Width, Reshape_True_Width))
            G = nx.Graph()
            G = nx.from_pandas_adjacency(df_adj)
            edges = G.edges()
            weights = [250*G[u][v]['weight'] for u,v in edges]
            nx.draw(G, ax=axs[(i//4)*2,i%4], width=weights)
            axs[(i//4)*2,i%4].title.set_text('Synchronizing')
            deg_seq = sorted((d for n, d in G.degree()), reverse=True)
            axs[(i//4)*2+1,i%4].plot(deg_seq, "b-", marker="o")

        for i in range(n):
            df_adj = pd.DataFrame(W_False[16:].T[i].reshape(Reshape_False_Width, Reshape_False_Width))
            G = nx.Graph()
            G = nx.from_pandas_adjacency(df_adj)
            edges = G.edges()
            weights = [250*G[u][v]['weight'] for u,v in edges]
            nx.draw(G, ax=axs[(i//4)*2,i%4+4], width=weights)
            axs[(i//4)*2,i%4+4].title.set_text('Non-Synchronizing')
            deg_seq = sorted((d for n, d in G.degree()), reverse=True)
            axs[(i//4)*2+1,i%4+4].plot(deg_seq, "b-", marker="o")
        if filename != None:
            fig.savefig(filename)
        plt.show()
        
    print(deg_seq)
        
n=8        
plot_adj_to_graph_deg(n, True)