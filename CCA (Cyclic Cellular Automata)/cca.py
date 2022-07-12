import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from networkx import grid_graph


def simulate_CCA(G, init_state, kappa, threshold, iters):

    """Implements the Cyclic Cellular Automata model

    Args:
        G (NetworkX Graph): Input graph to the model
        init_state (array): Initial state
        kappa (int): number of colors allowed
        threshold (int): allow a cell to be consumed when the # neighbors with successor value exceeds this threshold
        iters (int): number of iterations

    Returns:
        state_matrix: The dynamics of this CCA model with iters iterations;
    """

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

            #   CCA updateing rule (adpating from Wikiepdia):
                # The cyclic cellular automaton consists of a regular grid of cells in one or more dimensions. 
                # The cells can take on any of n states, ranging from 0 to n-1. 
                # The first generation starts out with random states in each of the cells. 
                # In each subsequent generation, if a cell has a neighboring cell whose value is the successor of the cell's value, 
                # the cell is "consumed" and takes on the succeeding value. 
                # (note that 0 is the successor of n-1

            for k in range(G.number_of_nodes()):
                # appliying von Neumann neighborhood (with edge directly conencted)
                if (curr_state[k] == (curr_state[j]+1) % kappa) and (list(G.nodes)[k] in list(G.adj[list(G.nodes)[j]])):
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
