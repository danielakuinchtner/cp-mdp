# COMPACT CP-MDP IMPLEMENTATION
# Daniela Kuinchtner

import os
import psutil
import time
import numpy as _np
import string
from utils.random_config import *
from utils.stpm import *
from utils.tensor_components import *
from utils.print_policy import *
from pymdptoolbox.mdp import *
import random

# install openblas lib to improve even more the runtime: conda install -c anaconda openblas

shape = [3, 4]  # z, y, x
# shape = [2, 3, 4]
number_of_obstacles = 1
number_of_terminals = 2
rewards_terminal_states = [100, -100]
reward_non_terminal_states = -3
p_intended = 0.8  # probability of the going to the intended state
discount = 0.9  # discount factor

print("Executing a", shape, "grid, with", number_of_terminals, "terminals and", number_of_obstacles, "obstacles")

start_time_precompute = time.time()
start_time_all = time.time()

states = 1
for dim in shape:
    states *= dim
print('Number of states: ', states)

dimensions = len(shape)
print('Number of dimensions: ', dimensions)

actions = _np.ones(len(shape) * 2)
letters_actions = []
acts = ['N', 'S', 'W', 'E', 'B', 'F']  # North, South, West, East, Backward, Forward
for num_actions in range(len(actions)):
    if len(actions) < 7:
        letters_actions.append(acts[num_actions])
    else:
        letters_actions.append(random.choice(string.ascii_letters))
print("Actions: ", letters_actions)

final_limits = []
for num_dim in range(len(shape)):
    new_shape = shape[num_dim] - 1
    final_limits.append(new_shape)
print('Final limits: ', final_limits)


# OBSTACLES AND TERMINALS:
# Option 1: Randomly placed obstacle and terminal states:
obstacles, terminals = randomConfig(number_of_obstacles, number_of_terminals, states)
obstacles = obstacles.astype(int)
terminals = terminals.astype(int)

# Option 2: manually placed obstacle and terminal states:
# obstacles = [5]
# terminals = [3, 7]

print("Obstacle states:", obstacles)
print("Terminal states:", terminals)


rewards = []
for num_term in range(number_of_terminals):
    for num_rew in range(len(rewards_terminal_states)):
        rewards.append(rewards_terminal_states[num_rew])
del rewards[number_of_terminals:]
print("Rewards for terminal states: ", rewards)

R = _np.full([states], reward_non_terminal_states)

for i in range(len(terminals)):
    R[terminals[i]] = rewards[i]
print("Rewards: ", R)


p_right_angles = (1 - p_intended) / (len(actions) - 2)  # 0.1 # stochasticity
p_opposite_angle = 0.0  # zero probability

STPM = mdp_stpm(p_intended, actions, p_right_angles, p_opposite_angle)
print("STPM: \n", STPM)

start_time_succ_vi = time.time()
start_time_succ = time.time()
succ_s, probability_s = tensorComponents(shape=shape, obstacles=obstacles, terminals=terminals, final_limits=final_limits,
                                                 STPM=STPM, states=states)
print("--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))

succ_s = _np.asarray(succ_s)
probability_s = _np.asarray(probability_s)
succ_s = _np.split(succ_s, len(STPM[0]))
probability_s = _np.split(probability_s, len(STPM[0]))
# Print tensor components:
# print("Probabilities: ", probability_s)
# print("Successors: ", succ_s)

start_time_vi = time.time()
vigs = CpMdpValueIterationGS(shape, terminals, obstacles, succ_s, probability_s, R, states, discount=discount, epsilon=0.001, max_iter=1000)
vigs.run()
print("--- Solved with CP-MDP-VI in: %s seconds ---" % (time.time() - start_time_vi))
process = psutil.Process(os.getpid())
print("Memory used CP-MDP-VI:", (process.memory_info().rss)/1000000, "Mb")

print("Policy CP-MDP-VI:")
printPolicy(vigs.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)

start_time_pi = time.time()
pi = CpMdpPolicyIteration(shape, terminals, obstacles, succ_s, probability_s, R, states, discount=discount, epsilon=0.001,policy0=None, max_iter=1000)
pi.run()
print("--- Solved with CP-MDP-PI in: %s seconds ---" % (time.time() - start_time_pi))
process = psutil.Process(os.getpid())
print("Memory used CP-MDP-PI:", (process.memory_info().rss)/1000000, "Mb")

print("Policy CP-MDP-PI:")
printPolicy(pi.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)


