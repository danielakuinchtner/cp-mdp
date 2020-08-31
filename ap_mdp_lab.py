# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
sys.path.insert(1,'pymdptoolbox/src')
import mdptoolbox.example
import time
from gen_scenario import *

"""
(Y,X)
| 00 01 02 ... 0X-1       'N' = North
| 10  .         .         'S' = South
| 20    .       .         'W' = West
| .       .     .         'E' = East
| .         .   .         'T' = Terminal
| .           . .         'O' = Obstacle
| Y-1,0 . . .   Y-1X-1
"""

shape = [3, 4]
rewards = [[0, 3, 100], [1, 3, -100]]
obstacles = [[1, 1]]
terminals = [[0, 3], [1, 3]]
actions = ['N', 'E', 'W', 'S']
final_limits = [shape[0]-1, shape[1]-1]
states = shape[0] * shape[1]

#start_time1 = time.time()
succ_xy, origin_xy, probability_xy, R = mdp_grid(shape=shape, terminals=terminals, r=-3,
                                                                         rewards=rewards, obstacles=obstacles,
                                                                         actions=actions, final_limits=final_limits)
#print("--- %s seconds ---" % (time.time() - start_time1))

#start_time2 = time.time()
vi = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_xy, origin_xy, probability_xy, R, states, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)
#print("--- %s seconds ---" % (time.time() - start_time2))

#--- 0.19688653945922852 seconds ---
#--- 0.0009992122650146484 seconds ---


vi.run()  # You can check the quadrant values using print vi.V
# print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, actions=actions)
# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)

# print(output)
# print(R)
