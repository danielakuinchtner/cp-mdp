# Randomly placed obstacle and terminal states

import numpy as _np
import random

def randomConfig(number_of_obstacles, number_of_terminals, states):
    obstacles = _np.array([])
    terminals = _np.array([])

    for n_obst in range(number_of_obstacles):
        obstacles = _np.append(obstacles, random.randint(0, states))

    for n_term in range(number_of_terminals):
        terminals = _np.append(terminals, random.randint(0, states))
    return obstacles, terminals