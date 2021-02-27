# Print policy

import numpy as _np


def printPolicy(policy, shape, obstacles=[], terminals=[], actions=[]):
    p_policy = _np.empty(shape, dtype=object)
    #print("terminals index", terminals)
    #print("obstacles index", obstacles)
    obstacles_tuple = []
    terminals_tuple = []

    for o in range(len(obstacles)):
        obstacles_tuple.append(_np.unravel_index(obstacles[o], shape))  # ind to sub

    for t in range(len(terminals)):
        terminals_tuple.append(_np.unravel_index(terminals[t], shape))  # ind to sub

    #print("terminals tuple", terminals_tuple)
    #print("obstacles tuple", obstacles_tuple)

    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)  # ind to sub
        if sub in obstacles_tuple:
            p_policy[sub] = 'O'
        elif sub in terminals_tuple:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)