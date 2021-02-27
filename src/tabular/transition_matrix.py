import numpy as _np
import math


def mdp_grid(shape=[], obstacles=[], terminals=[], r=1, rewards=[], final_limits=[], states=[], actions=[], STPM=[]):

    S = states
    A = actions
    P = _np.zeros([len(A),S,S])

    for A in range(len(STPM)):
        for s in range(S):
            for aa in range(len(STPM[A])):
                state_tuple = _np.unravel_index(s, shape)  # ind to sub
                state_tuple = list(state_tuple)
                if state_tuple in obstacles:
                    continue

                if state_tuple in terminals:
                    continue

                successor_state_of_s = succ_tuple(aa, state_tuple, final_limits)

                if successor_state_of_s not in obstacles:
                    state_to = _np.ravel_multi_index(successor_state_of_s, shape)  # sub to ind
                    state_to = state_to.item()
                    P[A, s, state_to] = P[A, s, state_to] + STPM[A][aa]
                else:
                    P[A, s, s] = P[A, s, s] + STPM[A][aa]

    R = _np.ones([S])
    R = _np.multiply(R, r)
    for i in range(len(terminals)):
        ind_terminal = _np.ravel_multi_index(terminals[i], shape)
        R[ind_terminal] = rewards[i]

    return P, R


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
    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)
        if list(sub) in obstacles:
            p_policy[sub] = 'O'
        elif list(sub) in terminals:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)


