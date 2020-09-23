import numpy as _np
import math
#import tensorflow as tf  # .\venv\Scripts\activate
import scipy.sparse as _sp
import sys

sys.path.insert(1, 'pymdptoolbox/src')
import mdptoolbox.example

"""
ACTIONS = ['N','S','E','W']
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


def mdp_grid(shape=[], obstacles=[], terminals=[], r=1, rewards=[], actions=[], final_limits=[]):
    actions = actions
    r = r
    terminals = terminals
    obstacles = obstacles
    final_limits = final_limits  # 2x3

    p_intended = 0.8  # probability of the desired action taking place
    p_right_angles = (1 - p_intended) / 2  # 0.1 # stochasticity
    p_opposite_angle = 0.0  # zero probability

    # State Transition Probability Matrix
    # N                E                 W                 S
    STPM = [[p_intended, p_right_angles, p_right_angles, p_opposite_angle],  # North
            [p_right_angles, p_intended, p_opposite_angle, p_right_angles],  # East
            [p_right_angles, p_opposite_angle, p_intended, p_right_angles],  # West
            [p_opposite_angle, p_right_angles, p_right_angles, p_intended]]  # South

    # STPM = [[0.8(N,N), 0.1(N,E), 0.1(N,W), 0.0(N,S)],
    #         [0.1(E,N), 0.8(E,E), 0.0(E,W), 0.1(E,S)],
    #         [0.1(W,N), 0.0(W,E), 0.8(W,W), 0.1(W,S)],
    #         [0.0(S,N), 0.1(S,E), 0.1(S,W), 0.8(S,S)]]

    successors = _np.array([])
    origins = _np.array([])
    probabilities = _np.array([])

    num_states = 1
    for value in shape:
        num_states *= value

    for a in STPM:
        for s in range(num_states):
            # for d in range(len(shape)):
            for aa in range(len(a)):  # 4
                if a[aa] == 0:  # remove all zero probabilities
                    continue

                state_tuple = _np.unravel_index(s, shape)  # ind to sub
                state_tuple = list(state_tuple)
                successor_state_of_s = succ_tuple(aa, state_tuple, final_limits)

                if successor_state_of_s not in obstacles:
                    state_to = _np.ravel_multi_index(successor_state_of_s, shape)  # sub to ind

                    if state_tuple in terminals or state_tuple in obstacles:
                        successors = _np.append(successors, s)
                        origins = _np.append(origins, s)
                        probabilities = _np.append(probabilities, 0)

                    else:
                        successors = _np.append(successors, state_to)
                        origins = _np.append(origins, s)
                        probabilities = _np.append(probabilities, a[aa])

                else:
                    successors = _np.append(successors, s)
                    origins = _np.append(origins, s)
                    probabilities = _np.append(probabilities, a[aa])


    successors = successors.astype(int)
    succ_xy = _np.split(successors, len(STPM))
    #print("succ", succ_xy)

    origins = origins.astype(int)
    origin_xy = _np.split(origins, len(STPM))
    #print("origin", origin_xy)

    probability_xy = _np.split(probabilities, len(STPM))
    print("prob", probability_xy)

    # convert all splits into a tensor
    #split_tensor = tf.convert_to_tensor(list(split), dtype=tf.float32)
    #split_tensor_origin_xy = tf.convert_to_tensor(list(origin_xy), dtype=tf.int32)
    #split_tensor_succ_xy = tf.convert_to_tensor(list(succ_xy), dtype=tf.int32)
    #dimensions = split_tensor.ndim
    #print(split_tensor)
    #print("Type of every element:", split_tensor.dtype)  # float32
    #print("Number of dimensions:", split_tensor.ndim)  # 3
    #print("Shape of tensor:", split_tensor.shape)  # (4, 27, 3)
    #print("Elements along axis 0 of tensor:", split_tensor.shape[0])  # 4
    #print("Elements along the last axis of tensor:", split_tensor.shape[-1])  # 3
    #print("Total number of elements (3*2*4*5): ", tf.size(split_tensor).numpy())  # 324

    R = _np.ones([num_states])
    R = _np.multiply(R, r)

    for i in range(len(terminals)):
        ind_terminal = _np.ravel_multi_index(terminals[i], shape)
        R[ind_terminal] = rewards[i]
    print(R)

    return succ_xy, origin_xy, probability_xy, R


def succ_tuple(a, state_tuple, final_limits):

    successor = []
    for dim in range(len(state_tuple)):

        if a == 0:  # North
            if state_tuple[dim] != 0:  # 1: limite inicial de X:0
                D = state_tuple[dim] - 1
            else:
                D = state_tuple[dim]

        if a == 1:  # East
            if state_tuple[dim] != final_limits[dim]:  # 4: limite final de Y:3
                D = state_tuple[dim] + 1
            else:
                D = state_tuple[dim]

        if a == 2:  # West
            if state_tuple[dim] != 0:  # 1: limite inicial de Y:0
                D = state_tuple[dim] - 1
            else:
                D = state_tuple[dim]

        if a == 3:  # South
            if state_tuple[dim] != final_limits[dim]:  # 3 limite inicial de X: 2
                D = state_tuple[dim] + 1
            else:
                D = state_tuple[dim]

        successor.append(D)
    return successor


def print_policy(policy, shape, obstacles=[], terminals=[], actions=[]):
    p_policy = _np.empty(shape, dtype=object)

    for i in range(len(policy)):
        sub = _np.unravel_index(i, shape)  # ind to sub
        if list(sub) in obstacles:
            p_policy[sub] = 'O'
        elif list(sub) in terminals:
            p_policy[sub] = 'T'
        else:
            p_policy[sub] = actions[policy[i]]
    print(p_policy)


def succ(a, state_pair, final_limits):

    if a == 0:  # North
        if state_pair[0] != 0:  # 1: limite inicial de X:0
            D = [state_pair[0] - 1, state_pair[1]]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 1:  # East
        if state_pair[1] != final_limits[1]:  # 4: limite final de Y:3
            D = [state_pair[0], state_pair[1] + 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 2:  # West
        if state_pair[1] != 0:  # 1: limite inicial de Y:0
            D = [state_pair[0], state_pair[1] - 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 3:  # South
        if state_pair[0] != final_limits[0]:  # 3 limite inicial de X: 2
            D = [state_pair[0] + 1, state_pair[1]]
        else:
            D = [state_pair[0], state_pair[1]]

    return D[0], D[1]

"""
def index_to_coords(index, shape):
    '''convert index to coordinates given the shape'''

    for i in range(len(shape)):
        divisor = int(_np.product(shape[i:]))
        value = index[i]//divisor
        #coords.append(value)
        index[i] -= value * divisor
    print(index)
    return index
    
    
def r_to_rss(P, R, terminals, obstacles):
     RSS = _np.zeros([4,len(P[1]),len(P[1])])
     for A in range(4):
         for I in range(len(P[1])):
             for J in range(len(P[1])):
                 if([I,J] in terminals):
                     RSS[A,I,J] = R[J]
                 if([I,J] in obstacles): RSS[A,I,J] = 0
                 RSS[A,I,J] = (P[A,I,J] * R[J])
     return RSS


def r_to_rs(P, R, terminals, shape):
    RS = _np.zeros([len(P[1]), 4])
    for I in range(len(P[1])):
        for A in range(4):
            sub = ind2sub(shape, I)
            # if sub in obstacles: RS[I,A] = 0
            if sub in terminals:
                RS[I, A] = R[I]
            else:
                for J in range(len(P[1])):
                    RS[I, A] = RS[I, A] + (P[A, I, J] * R[J])
    return RS


def sub2ind(shape, rows, cols):
    return rows * shape[1] + cols


def ind2sub(shape, ind):

    rows = int((ind / shape[1]))
    cols = int((ind % shape[1]))
    return [rows, cols]


def valid(x, y, obstacles):
    # valid = ((I >= 0) and (I < shape[0])) and ((J >= 0) and (J < shape[1]))
    valid = not [x, y] in obstacles
    return valid


def succ8actions(a, state_pair, final_limits):
    if a == 0:  # North
        if state_pair[0] != 0:  # 1: limite inicial de X:0
            D = [state_pair[0] - 1, state_pair[1]]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 1:  # East
        if state_pair[1] != final_limits[1]:  # 4: limite final de Y:3
            D = [state_pair[0], state_pair[1] + 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 2:  # West
        if state_pair[1] != 0:  # 1: limite inicial de Y:0
            D = [state_pair[0], state_pair[1] - 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 3:  # South
        if state_pair[0] != final_limits[0]:  # 3 limite inicial de X: 2
            D = [state_pair[0] + 1, state_pair[1]]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 4:  # northeast
        if state_pair[0] != 0 and state_pair[1] != final_limits[1]:  #
            D = [state_pair[0] - 1, state_pair[1] + 1]
        else:
            D = [state_pair[0], state_pair[1]]
    if a == 5:  # southeast
        if state_pair[0] != final_limits[0] and state_pair[1] != final_limits[1]:  #
            D = [state_pair[0] + 1, state_pair[1] + 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 6:  # northwest
        if state_pair[0] != 0 and state_pair[1] != 0:  #
            D = [state_pair[0] - 1, state_pair[1] - 1]
        else:
            D = [state_pair[0], state_pair[1]]

    if a == 7:  # southwest
        if state_pair[1] != 0 and state_pair[0] != final_limits[0]:  #
            D = [state_pair[0] + 1, state_pair[1] - 1]
        else:
            D = [state_pair[0], state_pair[1]]


    return D[0], D[1]
"""


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
