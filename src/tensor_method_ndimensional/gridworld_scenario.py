import numpy as _np
import math
import scipy.sparse as _sp
import sys
sys.path.insert(1, 'pymdptoolbox/src')
import mdptoolbox.example


def mdp_grid(shape=[], obstacles=[], terminals=[], reward_non_terminal_states=1, rewards=[], final_limits=[],
             STPM=[], states=[]):
    r = reward_non_terminal_states
    num_states = states
    terminals = terminals
    obstacles = obstacles
    final_limits = final_limits  # 2x3
    STPM = STPM

    successors = []
    probabilities = []

    for a in range(len(STPM)):
        for s in range(num_states):
            for aa in range(len(STPM[a])):
                if STPM[a][aa] == 0:  # remove all zero probabilities
                    continue

                if s in terminals: #or s in obstacles:
                    probabilities.append(0)
                    successors.append(s)

                else:
                    state_tuple = _np.unravel_index(s, shape)  # ind to sub
                    state_tuple = list(state_tuple)

                    successor_state_of_s = succ_tuple(aa, state_tuple, final_limits)

                    state_to = _np.ravel_multi_index(successor_state_of_s, shape)  # sub to ind
                    state_to = state_to.item()

                    if state_to not in obstacles:
                        successors.append(state_to)
                        probabilities.append(STPM[a][aa])

                    else:
                        successors.append(s)
                        probabilities.append(STPM[a][aa])

    R = _np.full([num_states], r)

    for i in range(len(terminals)):
        R[terminals[i]] = rewards[i]
    #print("Rewards:", R)


    return successors, probabilities, R


def succ_tuple(a, state_tuple, final_limits):

    successor = []
    for dim in range(len(state_tuple)):

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


def print_policy(policy, shape, obstacles=[], terminals=[], actions=[]):
    p_policy = _np.empty(shape, dtype=object)
    print("terminals index", terminals)
    print("obstacles index", obstacles)
    obstacles_tuple = []
    terminals_tuple = []

    for o in range(len(obstacles)):
        obstacles_tuple.append(_np.unravel_index(obstacles[o], shape))  # ind to sub

    for t in range(len(terminals)):
        terminals_tuple.append(_np.unravel_index(terminals[t], shape))  # ind to sub

    print("terminals tuple", terminals_tuple)
    print("obstacles tuple", obstacles_tuple)

    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)  # ind to sub
        if sub in obstacles_tuple:
            p_policy[sub] = 'O'
        elif sub in terminals_tuple:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)


