import sys
import os
import psutil

sys.path.insert(1, 'pymdptoolbox/src')
import time
import numpy as _np
import string
import random
# import tensorflow as tf
import math
import scipy.sparse as _sp


import tensorflow as tf

tf.test.gpu_device_name()

import torch


# import torch
# import pycuda.driver as cuda
# cuda.init()
# print(torch.cuda.current_device())
# print(cuda.Device(0).name()) # '0' is the id of your GPU
# print(torch.cuda.get_device_name(0))
# available, total = cuda.mem_get_info()
# print("Available: %.2f GB\nTotal:     %.2f GB"%(available/1e9, total/1e9))
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_cached())
# torch.cuda.empty_cache()

# GEN_SCENARIO:

def mdp_grid(shape=[], obstacles=[], terminals=[], reward_non_terminal_states=1, rewards=[], final_limits=[],
             STPM=[], states=[]):
    r = reward_non_terminal_states
    num_states = states
    terminals = terminals
    obstacles = obstacles
    final_limits = final_limits  # 2x3
    STPM = STPM

    # successors = _np.array([])
    # origins = _np.array([])
    # probabilities = _np.array([])
    output = []

    successors = []
    origins = []
    probabilities = []

    for a in range(len(STPM)):
        for s in range(num_states):
            for aa in range(len(STPM[a])):
                if STPM[a][aa] == 0:  # remove all zero probabilities
                    continue

                state_tuple = _np.unravel_index(s, shape)  # ind to sub
                state_tuple = list(state_tuple)

                successor_state_of_s = succ_tuple(aa, state_tuple, final_limits)
                # print(state_tuple, successor_state_of_s)

                if successor_state_of_s not in obstacles:
                    state_to = _np.ravel_multi_index(successor_state_of_s, shape)  # sub to ind
                    state_to = state_to.item()

                    if state_tuple in terminals or state_tuple in obstacles:
                        # successors = _np.append(successors, s)
                        # origins = _np.append(origins, s)
                        # probabilities = _np.append(probabilities, 0)

                        # origins.append(s)
                        probabilities.append(0)
                        successors.append(s)
                        # output.append([s, s, 0])

                    else:
                        # successors = _np.append(successors, state_to)
                        # origins = _np.append(origins, s)
                        # probabilities = _np.append(probabilities, a[aa])
                        # output.append([s, state_to, STPM[a][aa]])
                        successors.append(state_to)
                        # origins.append(s)
                        probabilities.append(STPM[a][aa])

                else:
                    # successors = _np.append(successors, s)
                    # origins = _np.append(origins, s)
                    # probabilities = _np.append(probabilities, a[aa])
                    # output.append([s, s, STPM[a][aa]])
                    successors.append(s)
                    # origins.append(s)
                    probabilities.append(STPM[a][aa])

    successors = torch.cuda.LongTensor(successors)
    # successors = torch.cuda.FloatTensor(successors)
    probabilities = torch.cuda.FloatTensor(probabilities)
    # print(successors.shape)
    # print(probabilities, successors)

    # print(type(successors))

    R = _np.ones([num_states])
    R = _np.multiply(R, r)
    # print(R)

    for i in range(len(terminals)):
        ind_terminal = _np.ravel_multi_index(terminals[i], shape)
        R[ind_terminal] = rewards[i]
    # print("Rewards:", R)

    # print(probabilities)
    # print(output)
    # successors = successors.astype(int)
    # succ_xy = _np.split(successors, len(STPM))
    # print("succ", succ_xy)

    # origins = origins.astype(int)
    # origin_xy = _np.split(origins, len(STPM))
    # print("origin", origin_xy)

    # probability_xy = _np.split(probabilities, len(STPM))
    # print("prob", probability_xy)

    return successors, probabilities, R


def succ_tuple(a, state_tuple, final_limits):
    successor = []
    for dim in prange(len(state_tuple)):

        if a - math.ceil(a / 2) == dim:
            if a % 2 == 0:
                if state_tuple[dim] != 0:
                    D = state_tuple[dim] - 1
                else:
                    D = state_tuple[dim]
            else:
                if state_tuple[dim] != final_limits[dim]:
                    D = state_tuple[dim] + 1
                else:
                    D = state_tuple[dim]
        else:
            D = state_tuple[dim]

        successor.append(D)

    return successor


def print_policy(policy, shape, obstacles=[], terminals=[], letters_actions=[]):
    p_policy = _np.empty(shape, dtype=object)
    actions = letters_actions
    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)  # ind to sub
        if list(sub) in obstacles:
            p_policy[sub] = 'O'
        elif list(sub) in terminals:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)