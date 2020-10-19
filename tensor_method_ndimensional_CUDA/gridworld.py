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
from numba import njit, prange


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

shape = [3, 4]
#shape = [3, 2, 3, 4]
number_of_obstacles = 1
number_of_terminals = 2
rewards = [100, -100]#, 100, -100]
reward_non_terminal_states = -3
p_intended = 0.8  # probability of the desired action taking place

print("Executing a", shape, "grid, with", number_of_terminals, "terminals and", number_of_obstacles, "obstacles")

start_time_precompute = time.time()
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

# State Transition Probability Matrix
#       N                S                 W                 E
# STPM = [[p_intended, p_opposite_angle, p_right_angles, p_right_angles],  # North
#        [p_opposite_angle, p_intended, p_right_angles, p_right_angles],  # East
#        [p_right_angles, p_right_angles, p_intended, p_opposite_angle],  # West
#        [p_right_angles, p_right_angles, p_opposite_angle, p_intended]]  # South

# STPM = [[0.8(N,N), 0.1(N,E), 0.1(N,W), 0.0(N,S)],
#         [0.1(E,N), 0.8(E,E), 0.0(E,W), 0.1(E,S)],
#         [0.1(W,N), 0.0(W,E), 0.8(W,W), 0.1(W,S)],
#         [0.0(S,N), 0.1(S,E), 0.1(S,W), 0.8(S,S)]]

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

print("--- Precomputed actions, obstacles and terminals in: %s seconds ---" % (time.time() - start_time_precompute))

#threads_per_block = 512
#blocks_per_grid = 36
#gpu_STPM = cuda.to_device(STPM)

start_time_succ_vi = time.time()
start_time_succ = time.time()


succ_xy, probability_xy, R = mdp_grid(shape=shape, terminals=terminals,
                                                 reward_non_terminal_states=reward_non_terminal_states,
                                                 rewards=rewards, obstacles=obstacles, final_limits=final_limits,
                                                 STPM=STPM, states=states)
#gpu_STPM.to_host()
print("--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))

#print(succ_xy, probability_xy)

"""
split_succ = tf.split(succ_xy, len(STPM), axis=0, num=None, name='split')
split_succ_tensor = tf.convert_to_tensor(list(split_succ), dtype=tf.float32)

split_probability = tf.split(probability_xy, len(STPM), axis=0, num=None, name='split')
split_probability_tensor = tf.convert_to_tensor(list(split_probability), dtype=tf.float32)

print(split_probability_tensor)
print(split_succ_tensor)
print("\nShape of tensor:", split_probability_tensor.shape)  # (4, 27, 3)
print("Total number of elements (3*2*4*5): ", tf.size(split_probability_tensor).numpy())  # 324

print("\nShape of tensor:", split_succ_tensor.shape)  # (4, 27, 3)
print("Total number of elements (3*2*4*5): ", tf.size(split_succ_tensor).numpy())  # 324
"""
succ_xy = _np.asarray(succ_xy)
probability_xy = _np.asarray(probability_xy)

#print(type(succ_xy))
#print(succ_xy)

succ_xy = _np.split(succ_xy, len(STPM[0]))
probability_xy = _np.split(probability_xy, len(STPM[0]))

#print(type(succ_xy))
#print(succ_xy)

start_time_vi = time.time()
#d_succ_xy = cuda.to_device(succ_xy)
#d_probability_xy = cuda.to_device(probability_xy)
vi = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_xy, probability_xy, R, states, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)
#d_succ_xy.to_host()
#d_probability_xy.to_host()
vi.run()
#d_succ_xy.to_host()
#d_probability_xy.to_host()

print("\n--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
print("\n--- Computed successors and rewards solved with VI in: %s seconds ---" % (time.time() - start_time_succ_vi))
print("\n--- Solved all in: %s seconds ---" % (time.time() - start_time_all))


process = psutil.Process(os.getpid())
print("\nMemory used:", (process.memory_info().rss), "bytes")
print("Memory used:", (process.memory_info().rss)/1000000, "Mb")
print("Memory used:", (process.memory_info().rss)/1000000000, "Gb")


#print("\nPolicy:")
print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, letters_actions=letters_actions)
# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)

