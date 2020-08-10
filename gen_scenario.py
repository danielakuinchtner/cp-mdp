import numpy as _np
# import tensorflow as tf
import scipy.sparse as _sp

# ACTIONS = ['N','S','E','W']


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


def mdp_grid(shape=[], obstacles=[], terminals=[], pm=0.8, r=1, rewards=[], actions=[], finalLimits=[]):
    S = shape[0] * shape[1]  # 3x4
    Actions = actions
    P = _np.zeros([len(Actions), S, S])
    ps = (1 - pm) / 2  # 0.1
    Terminals = terminals
    Obstacles = obstacles
    FinalLimits = finalLimits  # 2x3

    print(FinalLimits)
    print(Terminals)
    print(Obstacles)
    print(S)
    # validStates = S - len(Terminals) - len(Obstacles)
    # print(validStates)
    print(len(Actions))

    STPM =[[0.8, 0.1, 0.1, 0.0],  # N
          [0.1, 0.8, 0.0, 0.1],    # E
          [0.1, 0.0, 0.8, 0.1],  # W
          [0.0, 0.1, 0.1, 0.8]]  # S

    print(STPM[0])
    print(STPM[1])
    print(STPM[2])
    print(STPM[3])
    print(len(STPM))

    for a in STPM:
        for x in range(shape[0]):
            for y in range(shape[1]):
                if [x, y] in Obstacles or [x, y] in Terminals:
                    continue
                for probActionDestiny in range(len(a)):
                    if a[probActionDestiny] == 0:
                        continue

                    Sfrom = sub2ind(shape, x, y)  # 0,1,2,3,...,11,0,1,2,3,...,11,0,1,2,3,...,11,0,1,2,3,...,11
                    successor = succ(probActionDestiny, x, y, FinalLimits)
                    #print("a:", a, "aa:", a[probActionDestiny], "index aa:", probActionDestiny, " x:", x, " y:", y, " succ:", successor)

                    successori, successorj = succ(probActionDestiny, x, y, FinalLimits)

                    if valid(successori, successorj, Obstacles):    # se o succ não for obstáculo
                        Sto = sub2ind(shape, successori, successorj)
                        P[STPM.index(a), Sfrom, Sto] = P[STPM.index(a), Sfrom, Sto] + a[probActionDestiny]

                    else:                                           # se o succ for obstáculo
                        P[STPM.index(a), Sfrom, Sfrom] = P[STPM.index(a), Sfrom, Sfrom] + a[probActionDestiny]

                """
                    if valid(successori, successorj, Obstacles):        #se o succ não for obstáculo
                        Sto = sub2ind(shape, successori, successorj)
                        if a == aa:                                     #se A é a intenção desejada recebe 0.8
                            P[aa, Sfrom, Sto] = P[aa, Sfrom, Sto] + pm
                        else:                                           #senão 0.1
                            P[aa, Sfrom, Sto] = P[aa, Sfrom, Sto] + ps

                    else:                                               #se o succ for obstáculo
                        if a == aa:                                     #se A é a intenção desejada
                            P[aa, Sfrom, Sfrom] = P[aa, Sfrom, Sfrom] + pm
                        else:
                            P[aa, Sfrom, Sfrom] = P[aa, Sfrom, Sfrom] + ps

                    #Ca = tf.

                    """

                # ti, tj = front(A,I,J)

                # If the destination of the move is out of the grid, add Pm to self transition
                """
                if valid(ti,tj,shape,obstacles):
                    Sto = sub2ind(shape,ti,tj)
                    #print ("Front Sfrom ", Sfrom, "Sto ", Sto)
                    P[A,Sfrom,Sto] = pm;
                else:
                    P[A,Sfrom,Sfrom] = pm;
                
                #If any of the sides of the move are out of the grid, add Ps to self transition
                ti, tj = left(A,I,J);
                if valid(ti,tj,shape,obstacles):
                    Sto = sub2ind(shape,ti,tj)
                    P[A,Sfrom,Sto] = Ps
                else:
                    P[A,Sfrom,Sfrom] = P[A,Sfrom,Sfrom] + Ps
                
                ti, tj = right(A,I,J);
                if valid(ti,tj,shape,obstacles):
                    Sto = sub2ind(shape,ti,tj)
                    P[A,Sfrom,Sto] = Ps
                else:
                    P[A,Sfrom,Sfrom] = P[A,Sfrom,Sfrom] + Ps
                """

    R = _np.ones([S]);
    R = _np.multiply(R, r)
    for i in range(len(rewards)):
        Si = rewards[i][0]
        Sj = rewards[i][1]
        Sv = rewards[i][2]
        SR = sub2ind(shape, Si, Sj)
        R[SR] = Sv

    RSS = r_to_rs(P, R, terminals, obstacles, shape)
    return (P, RSS, R)


# def r_to_rss(P, R, terminals, obstacles):
#     RSS = _np.zeros([4,len(P[1]),len(P[1])])
#     for A in range(4):
#         for I in range(len(P[1])):
#             for J in range(len(P[1])):
#                 if([I,J] in terminals):
#                     RSS[A,I,J] = R[J]
#                 if([I,J] in obstacles): RSS[A,I,J] = 0
#                 RSS[A,I,J] = (P[A,I,J] * R[J])
#     return RSS

def r_to_rs(P, R, terminals, obstacles, shape):
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


def valid(x, y, Obstacles):
    # valid = ((I >= 0) and (I < shape[0])) and ((J >= 0) and (J < shape[1]))
    valid = not [x, y] in Obstacles
    return valid


def succ(a, x, y, FinalLimits):
    if a == 0:  # North
        # print(STPM)
        if x != 0:  # 1: limite inicial de X:0
            D = [x - 1, y]
        else:
            D = [x, y]

    if a == 1:  # East
        if y != FinalLimits[1]:  # 4: limite final de Y:3
            D = [x, y + 1]
        else:
            D = [x, y]

    if a == 2:  # West
        if y != 0:  # 1: limite inicial de Y:0
            D = [x, y - 1]
        else:
            D = [x, y]

    if a == 3:  # South
        if x != FinalLimits[0]:  # 3 limite inicial de X: 2
            D = [x + 1, y]
        else:
            D = [x, y]

    return D[0], D[1]


"""
#Returns the "left" position of the specified position given Action
def left(A, I, J):
    if A == 0:
        D =[I,J-1]
    elif A == 1: 
        D = [I,J+1]
    elif A == 2:
        D = [I-1,J]
    elif A == 3:
        D = [I+1,J]
    else:
        print("Invalid action")
        return 0,0
    return D[0], D[1]

#Returns the "right" position of the specified position given Action
def right(A, I, J):
    if A == 0:
        D = [I,J+1]
    elif A == 1:
        D = [I,J-1]
    elif A == 2:
        D = [I+1,J]
    elif A == 3:
        D = [I-1,J]
    else:
        print("Invalid action")
        return 0,0
    return D[0], D[1]


#Returns the "front" position of the specified position given Action
def front(A, I, J):
    if A == 0:
        D = [I-1,J]
    elif A == 1:
        D = [I+1,J]
    elif A == 2:
        D = [I,J+1]
    elif A == 3:
        D = [I,J-1]
    else:
        print("Invalid action")
        return 0,0
    return D[0], D[1]
"""


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
