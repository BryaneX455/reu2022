# assuming firefly module is in system path
#from firefly import * 
# assuming relative insertion of path
import sys
sys.path.insert(1,'../')
from firefly import *


# --------------------------- #
# Figure 2C (Bernoulli) - emission, reception and refractory
# --------------------------- #
edgelistcr = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6]] # parameter settings
colorlistcr = [3,0,1,2,3,4,5]; kcr = 6
netcr = ColorNNetwork(colorlistcr,edgelistcr) # construction and simulation
sync_display(simulate_FCA(netcr, kcr, its=6000, stochastic=0,
                          animate=True, name="star6", refractory=1)[0], kcr)


# --------------------------- #
# Figure 2C (Bernoulli) - reception
# --------------------------- #
edgelistcr = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6]] # parameter settings
colorlistcr = [3,0,1,2,3,4,5]; kcr = 6
netcr = ColorNNetwork(colorlistcr,edgelistcr) # construction and simulation
sync_display(simulate_FCA(netcr, kcr, its=6000, stochastic=1,
                          animate=True, name="star6")[0], kcr)


# --------------------------- #
# Figure 2C (Bernoulli) - emission
# --------------------------- #
edgelistcr = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6]] # parameter settings
colorlistcr = [3,0,1,2,3,4,5]; kcr = 6
netcr = ColorNNetwork(colorlistcr,edgelistcr) # construction and simulation
sync_display(simulate_FCA(netcr, kcr, its=6000, stochastic=2,
                          animate=True, name="star6")[0], kcr)


# --------------------------- #
# Figure 2C (Bernoulli) - emission, reception and refractory
# --------------------------- #
edgelistcr = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6]] # parameter settings
colorlistcr = [3,0,1,2,3,4,5]; kcr = 6
netcr = ColorNNetwork(colorlistcr,edgelistcr) # construction and simulation
sync_display(simulate_FCA(netcr, kcr, its=6000, stochastic=3,
                          animate=True, name="star6", refractory=1)[0], kcr)

