# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys

sys.path.insert(1, 'pymdptoolbox/src')
import mdptoolbox.example
import time
import numpy as _np
import string
import random
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

shape = [100, 100]
#shape = [2, 3, 4]
#shape = [3, 2, 3, 4]
number_of_obstacles = 20
number_of_terminals = 6
rewards = [100, -100, 100, -100, 100, -100]

states = 1
for dim in shape:
    states *= dim

dimensions = len(shape)
print('Number of dimensions: ', dimensions)

# actions = ['N', 'E', 'W', 'S', 'NE', 'SE', 'NW', 'SW']
# actions = ['N', 'S', 'W', 'E']
# actions = ['N', 'S', 'W', 'E', 'B', 'F']
# actions = ['N', 'S', 'W', 'E', 'B', 'F', 'C', 'D']


# deixar essa parte mais "esperta" (tirar letras O e T e não deixar repetir)
actions = _np.ones(len(shape) * 2)
letters_actions = []
for num_actions in actions:
    letters_actions.append(random.choice(string.ascii_letters))
print("Actions Letters: ", letters_actions)

final_limits = []
for num_dim in range(len(shape)):
    new_shape = shape[num_dim] - 1
    final_limits.append(new_shape)

def randomConfig():
    #obstacles = []
    #terminals = []
    obstacles = _np.array([])
    terminals = _np.array([])

    for n_obst in range(number_of_obstacles):
        for dim in range(len(shape)):
            #obstacles.append(random.randint(0, final_limits[dim]))
            obstacles = _np.append(obstacles, random.randint(0, final_limits[dim]))

    for n_term in range(number_of_terminals):
        for dim in range(len(shape)):
            terminals = _np.append(terminals, random.randint(0, final_limits[dim]))
            #terminals.append(random.randint(0, final_limits[dim]))
    return obstacles, terminals

obstacles, terminals = randomConfig()
obstacles = obstacles.astype(int)
terminals = terminals.astype(int)

obs = _np.split(obstacles, number_of_obstacles)
term = _np.split(terminals, number_of_terminals)

obstacles = []
for o in range(len(obs)):
    obstacles.append(obs[o].tolist())

terminals = []
for o in range(len(term)):
    terminals.append(term[o].tolist())

print("Obstacles:", obstacles)
print("Terminals:", terminals)


"""
obstacles = [[1, 1]]
terminals = [[0, 3], [1, 3]]
rewards = [100, -100]  # each reward corresponds to a terminal position

obstacles = [[0, 1, 1], [1, 1, 1]]
terminals = [[0, 0, 3], [0, 1, 3], [1, 0, 3], [1, 1, 3]]
rewards = [100, -100, 100, -100]  # each reward corresponds to a terminal position

obstacles = [[0, 0, 1, 1], [0, 1, 1, 1]]
terminals = [[0, 0, 0, 3], [0, 0, 1, 3], [0, 1, 0, 3], [0, 1, 1, 3]]
rewards = [100, -100, 100, -100]  # each reward corresponds to a terminal position
"""

print("Executing a", shape, "grid")

start_time1 = time.time()
succ_xy, origin_xy, probability_xy, R = mdp_grid(shape=shape, terminals=terminals, r=-3,
                                                 rewards=rewards, obstacles=obstacles,
                                                 actions=actions, final_limits=final_limits)

vi = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_xy, origin_xy, probability_xy, R, states,
                                     discount=1, epsilon=0.001, max_iter=1000, skip_check=True)

vi.run()

print("--- %s seconds ---" % (time.time() - start_time1))

print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, letters_actions=letters_actions)
# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)


# print(output)
# print(R)
