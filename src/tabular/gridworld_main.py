import sys
sys.path.insert(0, 'pymdptoolbox/src')
import time
from transition_matrix import *
import os
import psutil
import string
import random
import numpy as _np
from pymdptoolbox.mdp import *

shape = [3, 4]
number_of_obstacles = 1
number_of_terminals = 2
rewards_terminal_states = [100, -100]
reward_non_terminal_states = -3
p_intended = 0.8
discount = 0.9

print("Executing a", shape, "grid, with", number_of_terminals, "terminals and", number_of_obstacles, "obstacles")

start_time_precompute = time.time()
start_time_all = time.time()
states = 1
for dim in shape:
    states *= dim
print("Number of states: ", states)

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

# obstacles = [[1,1]]
# terminals = [[0,3], [1,3]]

print("Obstacles:", obstacles)
print("Terminals:", terminals)

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

# print(STPM)

# print("--- Precomputed actions, obstacles and terminals in: %s seconds ---" % (time.time() - start_time_precompute), "\n")

start_time_succ_vi = time.time()
start_time_succ = time.time()
P, R = mdp_grid(shape=shape, terminals=terminals, r=-3,
                rewards=rewards, obstacles=obstacles, final_limits=final_limits, states=states, actions=actions, STPM=STPM)

print("--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))

# print(P)
# print(R)


start_time_vi = time.time()
vigs = ValueIterationGS(P, R, discount=discount, epsilon=0.001, max_iter=1000, skip_check=True)
vigs.run()
print("--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
process = psutil.Process(os.getpid())
print("Memory used VI:", (process.memory_info().rss)/1000000, "Mb")


start_time_pi = time.time()
pi = PolicyIteration(P, R, discount=discount, policy0=None, max_iter=1000, eval_type=0, skip_check=True)
pi.run()
print("--- Solved with PI in: %s seconds ---" % (time.time() - start_time_pi))
process = psutil.Process(os.getpid())
print("Memory used PI:", (process.memory_info().rss)/1000000, "Mb")


print("Policy VI:")
print_policy(vigs.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)

print("\nPolicy PI:")
print_policy(pi.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)
