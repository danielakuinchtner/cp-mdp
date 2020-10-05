import numpy as _np
import scipy.sparse as _sp
import math

"""
input:
shape -> [Y,X] shape of the grid
obstacles -> [[Y1,X1], [Y2,X2]] list with the position of obstacles
terminals -> [[Y1,X1], [Y2,X2]] list with the position of terminal states
pm -> (0.0 - 1.0) the probability of successfully moving
r -> (double/int value) the default reward for all states
rewards -> [[Y1,X1,R1], [Y2,X2,R2]] a cell array of [Y,X,R] rewards for specific states

output: 
P = (A x S x S) the transition function
R = (A x S x S) the reward function
"""

def mdp_grid(shape=[], obstacles=[], terminals=[], r=1, rewards=[], final_limits=[], states=[], actions=[], STPM=[]):
    print("SHAPE:", shape)
    S = states
    A = actions
    print("S:", S)
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


def print_policy(policy, shape, obstacles=[], terminals=[], letters_actions=[]):
    p_policy = _np.empty(shape, dtype=object)
    actions = letters_actions
    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)
        if list(sub) in obstacles:
            p_policy[sub] = 'O'
        elif list(sub) in terminals:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)

from IPython.display import HTML, display

SYMBOLS = ['&uarr;','&darr;','&rarr;','&larr;']

def display_policy(policy, shape, obstacles=[], terminals=[]):
    p_policy = _np.empty(shape, dtype=object)
    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)
        if sub in obstacles: p_policy[sub[0]][sub[1]] = '&#x25FE;'
        elif sub in terminals: p_policy[sub[0]][sub[1]] = '&#x25CE;'
        else: p_policy[sub[0]][sub[1]] = SYMBOLS[policy[i]]
    display(HTML(
        '<table style="font-size:300%;border: thick solid;"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in p_policy)
            )
     ))