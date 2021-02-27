# State Transition Probability Matrix (STPM) generator

import numpy as _np

# State Transition Probability Matrix
#       N                S                 W                 E
# STPM = [[p_intended, p_opposite_angle, p_right_angles, p_right_angles],  # North
#        [p_opposite_angle, p_intended, p_right_angles, p_right_angles],  # East
#        [p_right_angles, p_right_angles, p_intended, p_opposite_angle],  # West
#        [p_right_angles, p_right_angles, p_opposite_angle, p_intended]]  # South

# STPM = [[0.8(N,N), 0.1(N,E), 0.1(N,W), 0.0(N,S)],
#         [0.1(E,N), 0.8(E,E), 0.0(E,W), 0.1(E,S)],
#         [0.1(W,N), 0.0(W,E), 0.8(W,W), 0.1(W,S)],
#         [0.0(S,N), 0.1(S,E), 0.1(S,W), 0.8(S,S)]]

def mdp_stpm(p_intended, actions, p_right_angles, p_opposite_angle):

    STPM = _np.ones([len(actions), len(actions)])
    STPM = _np.multiply(STPM, p_right_angles)

    for a1 in range(len(STPM[0])):
        for a2 in range(len(STPM[1])):
            if a1 == a2:
                STPM[a1, a2] = p_intended
                if a2 % 2 == 0:
                    STPM[a1, a2 + 1] = p_opposite_angle
                elif a2 % 2 == 1:
                    STPM[a1, a2 - 1] = p_opposite_angle

    return STPM
