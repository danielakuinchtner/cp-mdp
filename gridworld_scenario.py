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

    successors = _np.array([])
    origins = _np.array([])
    probabilities = _np.array([])
    output = []

    for a in STPM:
        for s in range(num_states):
            for aa in range(len(a)):  # 4
                if a[aa] == 0:  # remove all zero probabilities
                    continue

                state_tuple = _np.unravel_index(s, shape)  # ind to sub
                state_tuple = list(state_tuple)
                successor_state_of_s = succ_tuple(aa, state_tuple, final_limits)
                #print(state_tuple, successor_state_of_s)

                if successor_state_of_s not in obstacles:
                    state_to = _np.ravel_multi_index(successor_state_of_s, shape)  # sub to ind

                    if state_tuple in terminals or state_tuple in obstacles:
                        successors = _np.append(successors, s)
                        origins = _np.append(origins, s)
                        probabilities = _np.append(probabilities, 0)
                        output.append([s, s, 0])

                    else:
                        successors = _np.append(successors, state_to)
                        origins = _np.append(origins, s)
                        probabilities = _np.append(probabilities, a[aa])
                        output.append([state_to, s, a[aa]])

                else:
                    successors = _np.append(successors, s)
                    origins = _np.append(origins, s)
                    probabilities = _np.append(probabilities, a[aa])
                    output.append([s, s, a[aa]])

    R = _np.ones([num_states])
    R = _np.multiply(R, r)

    for i in range(len(terminals)):
        ind_terminal = _np.ravel_multi_index(terminals[i], shape)
        R[ind_terminal] = rewards[i]
    #print("Rewards:", R)

    successors = successors.astype(int)
    succ_xy = _np.split(successors, len(STPM))
    #print("succ", succ_xy)

    origins = origins.astype(int)
    origin_xy = _np.split(origins, len(STPM))
    #print("origin", origin_xy)

    probability_xy = _np.split(probabilities, len(STPM))
    #print("prob", probability_xy)

    return succ_xy, origin_xy, probability_xy, R, output


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
        sub = _np.unravel_index(i, shape)  # ind to sub
        if list(sub) in obstacles:
            p_policy[sub] = 'O'
        elif list(sub) in terminals:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)



from IPython.display import HTML, display

SYMBOLS = ['&uarr;', '&darr;', '&rarr;', '&larr;']


def display_policy(policy, shape, obstacles=[], terminals=[]):
    p_policy = _np.empty(shape, dtype=object)
    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)
        if sub in obstacles:
            p_policy[sub[0]][sub[1]] = '&#x25FE;'
        elif sub in terminals:
            p_policy[sub[0]][sub[1]] = '&#x25CE;'
        else:
            p_policy[sub[0]][sub[1]] = SYMBOLS[policy[i]]
    display(HTML(
        '<table style="font-size:300%;border: thick solid;"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in p_policy)
        )
    ))
