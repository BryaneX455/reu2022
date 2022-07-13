# assuming firefly module is in system path
#from firefly import * 
# assuming relative insertion of path
import sys
sys.path.insert(1,'../')
from firefly import *

# --------------------------- #
# Cycle 100
# --------------------------- #
edgelistcyc100 = edgeset_generator([10],type='cycle', show=False) # parameter settings
k100 = 4; colorlist10 = np.random.randint(4, size=10)
colorlist10 = [0,2,3,2,3,2,2,3,0,0] 
net100 = ColorNNetwork(colorlist10, edgelistcyc100) # construction and simulation
sync_display(simulate_FCA(net100, k100, its=100, animate=False, timesec=1200,
                          name="Cycle10")[0], k100)
# this does not always work! (for 20 colors)

# --------------------------- #
# Cycle 
# --------------------------- #
edgelistcyc5 = edgeset_generator([5],type='cycle', show=False) # parameter settings
k = 4; #colorlist5 = np.random.randint(4, size=5)
colorlist5 = [3,2,3,3,2] 
net5 = ColorNNetwork(colorlist5, edgelistcyc5) # construction and simulation
sync_display(simulate_FCA(net5, k, its=100, timesec=1200)[0], k)
# this does not always work! (for 20 colors)
