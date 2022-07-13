# assuming firefly module is in system path
#from firefly import * 
# assuming relative insertion of path
import sys
sys.path.insert(1,'../')
from firefly import *

# --------------------------- #
# Standard deviation dynamics
# --------------------------- #
edgelist400 = edgeset_generator([500], type="cycle")
kappa = 20; colorlist400 = np.random.randint(10, size=500)
net400 = ColorNNetwork(colorlist400,edgelist400)
stat_result = color_statistics(net400, kappa)
plt.plot(stat_result[0])
plt.plot(stat_result[1])
plt.ylabel('std deviation')
plt.xlabel('coloring')
plt.xticks(ticks=[])

# ---------------------------- #
# Width dynamics 
# ---------------------------- #
edgelist400 = edgeset_generator([100], type="cycle")
kappa = 15; colorlist400 = np.random.randint(4, size=100)
net400 = ColorNNetwork(colorlist400,edgelist400)
# generate the plot
plt.plot(width_dynamics(net400,kappa))
plt.ylabel('widths')
plt.xlabel('colorings')
plt.show()
