# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
sys.path.insert(0, 'pymdptoolbox/src')
import mdptoolbox.example
import time
from gen_scenario import *


"""
(Y,X)
| 00 01 02 ... 0X-1
| 10  .         .
| 20    .       .
| .       .     .
| .         .   .
| .           . .
| Y-1,0 . . .   Y-1X-1
"""


shape = [150,150]
rewards = [[0,3,100],[1,3,-100]]
obstacles = [[1,1]]
terminals = [[0,3],[1,3]]

start_time_all = time.time()
start_time_succ = time.time()
P, R = mdp_grid(shape=shape, terminals=terminals, r=-3, rewards=rewards, obstacles=obstacles)
print("\n--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))
start_time_vi = time.time()
vi = mdptoolbox.mdp.ValueIterationGS(P, R, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)

# vi.verbose = True # Uncomment this for question 2
vi.run()
print("\n--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
print("\n--- Solved all in: %s seconds ---" % (time.time() - start_time_all))