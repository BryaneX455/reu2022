import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from networkx import grid_graph


def simulate_CCA(G, init_state, kappa, threshold, iters):
    state_matrix, curr_state = init_state
    next_state = np.zeros(G.number_of_nodes())

    for i in range(iters):
        if i != 0:
            curr_state = next_state
            state_matrix = np.vstack((state_matrix, next_state))

        next_state = np.zeros(G.number_of_nodes())

        for j in range(G.number_of_nodes()):
            flag = False
            count = 0

            for k in range(G.number_of_nodes()):
                # von Neumann neighborhood
                if (curr_state[k] == (curr_state[j]+1) % kappa) and (list(G.nodes)[k] in list(G.adj[list(G.nodes)[i]])):
                    count += 1
                    if count >= threshold:
                        flag = True
                        break

            if flag:
                next_state[j] = (curr_state[j]+1) % kappa
            else:
                next_state[j] = curr_state[j]

    if len(np.unique(state_matrix[-1])) == 1 and iters != 1:
        print("Synchronized!")
    else:
        print("Non-synchronized")

    return state_matrix
