# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
sys.path.insert(1,'pymdptoolbox/src')
import mdptoolbox.example
import time
from gridworld_scenario import *

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

# install openblas lib to improve even more the runtime: conda install -c anaconda openblas

shape = [3, 4, 2]

states = 1
for value in shape:
    states *= value

dimensions = len(shape)
print('Number of dimensions: ', dimensions)

#actions = ['N', 'E', 'W', 'S', 'NE', 'SE', 'NW', 'SW']
actions = ['N', 'E', 'W', 'S']


def randomConfig():
    for s in range(states):
        rewards = ''
        terminals = ''
        obstacles = ''



rewards = [[0, 3, 0, 100], [1, 3, 0, -100]]
obstacles = [[1, 1, 1]]
terminals = [[0, 3, 0], [1, 3, 0]]

final_limits = []
for num_dim in range(len(shape)):
    new_shape = shape[num_dim]-1
    final_limits.append(new_shape)

print("Executing a", shape, "grid")

start_time1 = time.time()
succ_xy, origin_xy, probability_xy, R = mdp_grid(shape=shape, terminals=terminals, r=-3,
                                                                         rewards=rewards, obstacles=obstacles,
                                                                         actions=actions, final_limits=final_limits)

vi = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_xy, origin_xy, probability_xy, R, states,
                                     discount=1, epsilon=0.001, max_iter=1000, skip_check=True)

vi.run()

print("--- %s seconds ---" % (time.time() - start_time1))

print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, actions=actions)
# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)


# print(output)
# print(R)
