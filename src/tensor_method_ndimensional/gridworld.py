
import sys
import os
import psutil
sys.path.insert(1, 'pymdptoolbox/src')
import mdptoolbox
import time
import numpy as _np
import string
import random
from gridworld_scenario import *


# install openblas lib to improve even more the runtime: conda install -c anaconda openblas

shape = [3,4]
#shape = [3, 10, 3, 10]
number_of_obstacles = 1
number_of_terminals = 2
rewards = [100, -100, 100, -100, 100, -100, 100, -100, 100, -100, 100, -100, 100, -100]
reward_non_terminal_states = -3
p_intended = 0.8  # probability of the desired action taking place
discount = 0.9

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
#print("Actions: ", letters_actions)

final_limits = []
for num_dim in range(len(shape)):
    new_shape = shape[num_dim] - 1
    final_limits.append(new_shape)


def randomConfig():
    obstacles = _np.array([])
    terminals = _np.array([])

    for n_obst in range(number_of_obstacles):
        obstacles = _np.append(obstacles, random.randint(0, states))

    for n_term in range(number_of_terminals):
        terminals = _np.append(terminals, random.randint(0, states))
    return obstacles, terminals


obstacles, terminals = randomConfig()
obstacles = obstacles.astype(int)
terminals = terminals.astype(int)


obstacles = [5]
terminals = [3, 7]

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



#print("--- Precomputed actions, obstacles and terminals in: %s seconds ---" % (time.time() - start_time_precompute))

start_time_succ_vi = time.time()
start_time_succ = time.time()
succ_s, probability_s, R = mdp_grid(shape=shape, terminals=terminals,
                                                 reward_non_terminal_states=reward_non_terminal_states,
                                                 rewards=rewards, obstacles=obstacles, final_limits=final_limits,
                                                 STPM=STPM, states=states)
print("--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))



succ_s = _np.asarray(succ_s)
probability_s = _np.asarray(probability_s)

succ_s = _np.split(succ_s, len(STPM[0]))
probability_s = _np.split(probability_s, len(STPM[0]))
#print(succ_s)
"""
start_time_vi = time.time()
vi = mdptoolbox.mdp.ValueIteration(shape, terminals, obstacles, succ_s, probability_s, R, states,
                                     discount=discount, epsilon=0.001, max_iter=1000, skip_check=True)

vi.run()

print("--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
"""

start_time_vi = time.time()
vigs = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_s, probability_s, R, states, discount=discount, epsilon=0.001, max_iter=1000, skip_check=True)
vigs.run()
print("--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
process = psutil.Process(os.getpid())
print("Memory used VI:", (process.memory_info().rss)/1000000, "Mb")

start_time_pi = time.time()
pi = mdptoolbox.mdp.PolicyIteration(shape, terminals, obstacles, succ_s, probability_s, R, states, discount=discount, epsilon=0.001,policy0=None, max_iter=1000, eval_type=0, skip_check=True)
pi.run()
print("--- Solved with PI in: %s seconds ---" % (time.time() - start_time_pi))
process = psutil.Process(os.getpid())
print("Memory used PI:", (process.memory_info().rss)/1000000, "Mb")


"""
start_time_mpi = time.time()
mpi = mdptoolbox.mdp.PolicyIterationModified(shape, terminals, obstacles, succ_s_pi, probability_s_pi, R, states,
                                             discount=discount, epsilon=0.001, policy0=None, max_iter=1000,
                                             eval_type=0, skip_check=True)
mpi.run()
print("--- Solved with MPI in: %s seconds ---" % (time.time() - start_time_mpi))
process = psutil.Process(os.getpid())
print("Memory used MPI:", (process.memory_info().rss)/1000000, "Mb")

print("Policy MPI:")
print_policy(mpi.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)
"""


print("Policy VI:")
print_policy(vigs.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)

print("Policy PI:")
#print(pi.policy)
print_policy(pi.policy, shape, obstacles=obstacles, terminals=terminals, actions=letters_actions)


