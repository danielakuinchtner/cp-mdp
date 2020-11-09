# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
import os
import psutil
sys.path.insert(1, 'pymdptoolbox/src')
import mdptoolbox.example
import time
import numpy as _np
import string
import random
#import tensorflow as tf
from gridworld_scenario import *
import torch
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

shape = [50, 50]
#shape = [3, 2, 3, 4]
number_of_obstacles = 1
number_of_terminals = 2
rewards = [100, -100] #, 100, -100]
reward_non_terminal_states = -3
p_intended = 0.8  # probability of the desired action taking place

print("Executing a", shape, "grid, with", number_of_terminals, "terminals and", number_of_obstacles, "obstacles")

#start_time_precompute = time.time()
start_time_all = time.time()
states = 1
for dim in shape:
    states *= dim

dimensions = len(shape)
# print('Number of dimensions: ', dimensions)

actions = _np.ones(len(shape) * 2)
letters_actions = []
acts = ['N', 'S', 'W', 'E', 'B', 'F']  # North, South, West, East, Backward, Forward
for num_actions in range(len(actions)):
    if len(actions) < 7:
        letters_actions.append(acts[num_actions])
    else:
        letters_actions.append(random.choice(string.ascii_letters))
#print("Actions Letters: ", letters_actions)

final_limits = []
for num_dim in range(len(shape)):
    new_shape = shape[num_dim] - 1
    final_limits.append(new_shape)


def randomConfig():
    obstacles = _np.array([])
    terminals = _np.array([])

    for n_obst in range(number_of_obstacles):
        for dim in range(len(shape)):
            obstacles = _np.append(obstacles, random.randint(0, final_limits[dim]))

    for n_term in range(number_of_terminals):
        for dim in range(len(shape)):
            terminals = _np.append(terminals, random.randint(0, final_limits[dim]))
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


obstacles = [[1, 1]]
terminals = [[0, 3], [1, 3]]
rewards = [100, -100]  # each reward corresponds to a terminal position
"""
obstacles = [[0, 1, 1], [1, 1, 1]]
terminals = [[0, 0, 3], [0, 1, 3], [1, 0, 3], [1, 1, 3]]
rewards = [100, -100, 100, -100]  # each reward corresponds to a terminal position

obstacles = [[0, 0, 1, 1], [0, 1, 1, 1]]
terminals = [[0, 0, 0, 3], [0, 0, 1, 3], [0, 1, 0, 3], [0, 1, 1, 3]]
rewards = [100, -100, 100, -100]  # each reward corresponds to a terminal position
"""

# print("Obstacles:", obstacles)
# print("Terminals:", terminals)

p_right_angles = (1 - p_intended) / (len(actions) - 2)  # 0.1 # stochasticity
p_opposite_angle = 0.0  # zero probability


STPM = _np.ones([len(actions), len(actions)])
STPM = _np.multiply(STPM, p_right_angles)

for a1 in range(len(STPM[0])):
    for a2 in range(len(STPM[1])):
        if a1 == a2:
            STPM[a1, a2] = p_intended
            if a2 % 2 == 0:
                STPM[a1, a2 + 1] = p_opposite_angle
            elif a2 % 2 == 1:
                STPM[a1, a2 - 1] = p_opposite_angle

ind_terminals = []
for t in range(len(terminals)):
    ind_terminals.append(_np.ravel_multi_index(terminals[t], shape))
#print(ind_terminals)

#print("--- Precomputed actions, obstacles and terminals in: %s seconds ---" % (time.time() - start_time_precompute))

#start_time_succ_vi = time.time()
start_time_succ = time.time()

succ_s, probability_s, R = mdp_grid(shape=shape, terminals=terminals,
                                                 reward_non_terminal_states=reward_non_terminal_states,
                                                 rewards=rewards, obstacles=obstacles, final_limits=final_limits,
                                                 STPM=STPM, states=states)
print("--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))

#print(succ_s)
#print(probability_s)

#succ_s = _np.asarray(succ_s)
#probability_s = _np.asarray(probability_s)
#print(type(succ_s))
#print(succ_s)


act_valid = len(STPM[0])-1
div = act_valid*states
succ_s = torch.split(succ_s, div)
probability_s = torch.split(probability_s, div)


start_time_vi = time.time()
vi = ValueIterationGS(shape, terminals, obstacles, succ_s, probability_s, R, states, discount=1,
                                     epsilon=0.001, max_iter=1000, skip_check=True)

vi.run()

print("\n--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
#print("\n--- Computed successors and rewards solved with VI in: %s seconds ---" % (time.time() - start_time_succ_vi))
print("\n--- Solved all in: %s seconds ---" % (time.time() - start_time_all))


process = psutil.Process(os.getpid())
print("Memory used:", (process.memory_info().rss), "bytes")
print("Memory used:", (process.memory_info().rss)/1000000, "Mb")
print("Memory used:", (process.memory_info().rss)/1000000000, "Gb")


print("\nPolicy:")
print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, letters_actions=letters_actions)

from platform import python_version

print(python_version())


print(torch.cuda.is_available())

"""
start_time_pi = time.time()
pi = mdptoolbox.mdp.PolicyIteration(succ_xy, probability_xy, R, discount=1, policy0=None, max_iter=1000, eval_type=0, skip_check=True)
pi.run()
print("\n--- Solved with PI in: %s seconds ---" % (time.time() - start_time_pi))

start_time_mpi = time.time()
mpi = mdptoolbox.mdp.PolicyIterationModified(succ_xy, probability_xy, R, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)
mpi.run()
print("\n--- Solved with MPI in: %s seconds ---" % (time.time() - start_time_mpi))



# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)
print("\nPolicy PI:")
print_policy(pi.policy, shape, obstacles=obstacles, terminals=terminals)
print("\nPolicy MPI:")
print_policy(mpi.policy, shape, obstacles=obstacles, terminals=terminals)

"""