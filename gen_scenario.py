import numpy as _np
import math
import tensorflow as tf  # .\venv\Scripts\activate
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
    states = shape[0] * shape[1]  # 3x4
    actions = actions
    r = r
    # P = _np.zeros([len(actions), states, states])
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

    # Ca = [[],[],[]]
    output = []

    for a in STPM:  # 4
        for x in range(shape[0]):  # 3
            for y in range(shape[1]):  # 4
                if [x, y] in obstacles or [x, y] in terminals:  # remove obstacles and terminals
                    continue
                for aa in range(len(a)):  # 4
                    if a[aa] == 0:  # remove all zero probabilities
                        continue

                    state_from = sub2ind(shape, x, y)

                    # xy = [x, y]
                    # successor = succ(aa, x, y, final_limits)
                    # print("a:", a, "aa:", a[aa], "index aa:", aa, " x:", x, " y:", y, " succ:", successor)

                    successor_i, successor_j = succ(aa, x, y, final_limits)

                    # if valid(successor_i, successor_j, obstacles):
                    if [x, y] not in obstacles:
                        state_to = sub2ind(shape, successor_i, successor_j)
                        # P[STPM.index(a), state_from, state_to] = P[STPM.index(a), state_from, state_to] + a[aa]
                        # successor = ind2sub(shape, state_to)
                        # Ca = Ca + [xy, successor, a[aa]]

                        rank_3_tensor = tf.constant([
                            state_from,
                            state_to,
                            a[aa]])
                        output.append(rank_3_tensor)

                    else:
                        # P[STPM.index(a), state_from, state_from] = P[STPM.index(a), state_from, state_from] + a[aa]
                        # Ca = Ca + [xy, xy, a[aa]]

                        rank_3_tensor = tf.constant([
                            state_from,
                            state_from,
                            a[aa]])
                        output.append(rank_3_tensor)

    # split the output into 4 arrays (because there are 4 actions)
    split = tf.split(output, len(STPM), axis=0, num=None, name='split')
    # print(split)

    # convert all splits into a tensor
    split_tensor = tf.convert_to_tensor(list(split), dtype=tf.float32)

    dimensions = split_tensor.ndim

    print(split_tensor)
    # print(split_tensor[0][:, 2])

    print("Type of every element:", split_tensor.dtype)  # float32
    print("Number of dimensions:", split_tensor.ndim)  # 3
    print("Shape of tensor:", split_tensor.shape)  # (4, 27, 3)
    print("Elements along axis 0 of tensor:", split_tensor.shape[0])  # 4
    print("Elements along the last axis of tensor:", split_tensor.shape[-1])  # 3
    print("Total number of elements (3*2*4*5): ", tf.size(split_tensor).numpy())  # 324

    """
    R = _np.ones([states])
    R = _np.multiply(R, r)

    SR_output = []
    for i in range(len(rewards)):
        Si = rewards[i][0]  # 0 # 1
        Sj = rewards[i][1]  # 3 # 3
        Sv = rewards[i][2]  # 100 # -100
        SR = sub2ind(shape, Si, Sj)  # index 3 and index 7
        SR_output.append(SR)
        R[SR] = Sv  # 100 # -100

    # RSS = r_to_rs(P, R, terminals, shape)
    # print(RSS)
    """

    RW = reward_tensor(split_tensor, r, rewards, terminals, obstacles, shape)
    print(RW)

    return split_tensor, RW, dimensions


def reward_tensor(split_tensor, r, rewards, terminals, obstacles, shape):
    RS_tensor = _np.zeros([shape[0], shape[1]])

    for i in range(shape[0]):
        for j in range(shape[1]):
            if [i, j] in terminals:
                RS_tensor[i, j] = rewards[i][2]

            elif [i, j] in obstacles:
                RS_tensor[i, j] += r

            else:
                for k in split_tensor[0]:
                    pair_xy = ind2sub(shape, k[0])

                    if [i, j] == pair_xy:
                        RS_tensor[i, j] += k[2] * r

    return RS_tensor


"""
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
"""


def sub2ind(shape, rows, cols):
    return rows * shape[1] + cols


def ind2sub(shape, ind):
    rows = int((ind / shape[1]))
    cols = int((ind % shape[1]))
    return [rows, cols]


"""
def valid(x, y, obstacles):
    # valid = ((I >= 0) and (I < shape[0])) and ((J >= 0) and (J < shape[1]))
    valid = not [x, y] in obstacles
    return valid
"""


def succ(a, x, y, final_limits):
    if a == 0:  # North
        if x != 0:  # 1: limite inicial de X:0
            D = [x - 1, y]
        else:
            D = [x, y]

    if a == 1:  # East
        if y != final_limits[1]:  # 4: limite final de Y:3
            D = [x, y + 1]
        else:
            D = [x, y]

    if a == 2:  # West
        if y != 0:  # 1: limite inicial de Y:0
            D = [x, y - 1]
        else:
            D = [x, y]

    if a == 3:  # South
        if x != final_limits[0]:  # 3 limite inicial de X: 2
            D = [x + 1, y]
        else:
            D = [x, y]

    return D[0], D[1]


def print_policy(policy, shape, obstacles=[], terminals=[], actions=[]):
    p_policy = _np.empty(shape, dtype=object)
    for i in range(len(policy)):
        sub = ind2sub(shape, i)
        if sub in obstacles:
            p_policy[sub[0]][sub[1]] = 'O'
        elif sub in terminals:
            p_policy[sub[0]][sub[1]] = 'T'
        else:
            p_policy[sub[0]][sub[1]] = actions[policy[i]]
    print(p_policy)


from IPython.display import HTML, display

SYMBOLS = ['&uarr;', '&darr;', '&rarr;', '&larr;']


def display_policy(policy, shape, obstacles=[], terminals=[]):
    p_policy = _np.empty(shape, dtype=object)
    for i in range(len(policy)):
        sub = ind2sub(shape, i)
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
