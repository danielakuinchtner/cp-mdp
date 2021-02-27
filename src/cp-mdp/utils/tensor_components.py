# Tensor components generator

import numpy as _np
import math


def tensorComponents(shape=[], obstacles=[], terminals=[], final_limits=[], STPM=[], states=[]):
    num_states = states
    terminals = terminals
    obstacles = obstacles
    final_limits = final_limits
    STPM = STPM

    successors = []
    probabilities = []

    for a in range(len(STPM)):
        for s in range(num_states):
            for aa in range(len(STPM[a])):
                if STPM[a][aa] == 0:  # remove all zero probabilities
                    continue

                if s in terminals or s in obstacles:
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

    return successors, probabilities


def succ_tuple(a, state_tuple, final_limits):
    successor = []
    for dim in range(len(state_tuple)):

        if a - math.ceil(a / 2) == dim:
            if a % 2 == 0:
                if state_tuple[dim] != 0:
                    d = state_tuple[dim] - 1
                else:
                    d = state_tuple[dim]
            else:
                if state_tuple[dim] != final_limits[dim]:
                    d = state_tuple[dim] + 1
                else:
                    d = state_tuple[dim]
        else:
            d = state_tuple[dim]

        successor.append(d)

    return successor
