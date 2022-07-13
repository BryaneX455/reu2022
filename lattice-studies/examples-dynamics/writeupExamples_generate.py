# assuming firefly module is in system path
#from firefly import * 
# assuming relative insertion of path
import sys
sys.path.insert(1,'../')
from firefly import *

# --------------------------- #
# Figure 2A 
# --------------------------- #
edgelista = [[0,1]]; colorlista = [1,3]; ka = 6 # parameter settings
neta = ColorNNetwork(colorlista,edgelista) # construction and simulation
sync_display(simulate_FCA(neta, ka, animate=1, name='K2')[0], ka)



# --------------------------- #
# Figure 2B - Generates GIF
# --------------------------- #
edgelistb = [[0,1], [1,2], [2,3], [3,1], [3,4], [4,0]] # parameter settings
colorlistb = [1,2,1,3,4]; kb = 6
netb = ColorNNetwork(colorlistb, edgelistb) # construction and simulation
sync_display(simulate_FCA(netb,6,animate=True, name="V5")[0],
             kb, name="K2")


# --------------------------- #
# Figure 2C // no sync 
# --------------------------- #
edgelistc = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6]] # parameter settings
colorlistc = [3,0,1,2,3,4,5]; kc = 6
netc = ColorNNetwork(colorlistc, edgelistc) # construction and simulation
sync_display(simulate_FCA(netc, kc, its=6000)[0], kc)



# --------------------------- #
# Figure 2D // no sync 
# --------------------------- #
edgelistd = [[0,1],[0,2],[0,3],[1,3],[2,3],[1,2]] # parameter settings
colorlistd = [5,2,4,0]; kd = 6
netd = ColorNNetwork(colorlistd, edgelistd) # construction and simulation
sync_display(simulate_FCA(netd, kd, its=6000)[0], kd, name="Star6")



"""
Create path graph 20-color, save the sequence of colors over iterations
"""
edgelist400 = edgeset_generator([400,1],show=False) # parameter settings
k400 = 20; colorlist400 = np.random.randint(k400, size=400)
net400 = ColorNNetwork(colorlist400, edgelist400) # construction and simulation
sync_display(simulate_FCA(net400, k400, its=10000)[0], k400, name="P400")



"""
Create path graph 20-color, save the sequence of colors over iterations
"""
edgelist400 = edgeset_generator([400,1],show=False) # parameter settings
k400 = 20; colorlist400 = np.random.randint(k400, size=400)
net400 = ColorNNetwork(colorlist400, edgelist400) # construction and simulation
sync_display(simulate_FCA(net400, k400, its=10000)[0], k400, name="P400")



"""
Create Cycle graph 21-color, satisfying conentration condition, 
generates gif of actual graph and color labels 
"""
edgelistcyc100 = edgeset_generator([100],type='cycle', show=False) # parameter settings
k100 = 21; colorlist100 = np.random.randint(10, size=200)
net100 = ColorNNetwork(colorlist100, edgelistcyc100) # construction and simulation
sync_display(simulate_FCA(net100, k100, its=100000, animate=True, timesec=1200,
                          name="Cycle100")[0], k100)
# this does not always work! (for 20 colors)



