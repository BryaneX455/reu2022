import sys; sys.path.insert(1,'../')
from firefly import *

from numpy.random import randint
import numpy as np
import networkx as nx

n = 10 
edgelist = edgeset_generator([n,0.5], type="er")
kappa = 4; colorlist = randint(4, size=5)
graph = ColorNNetwork(colorlist, edgelist)

incoming = [None]*(graph.number_nodes)
#generate weight of each node
randlist = np.random.rand(graph.number_nodes)
#+1 if node labels start at 1

print(incoming)
print(tree_iter(graph, randlist, incoming))

# generate graphs
G = nx.Graph()
G.add_nodes_from(list(range(5)))
G.add_edges_from(graph.edges)
#pos = nx.spring_layout(G, seed=42)
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, list(range(5)), node_size=200)
nx.draw_networkx_edges(G, pos)
plt.show()
print(len(graph.edges))
