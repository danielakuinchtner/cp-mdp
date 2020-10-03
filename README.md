# A Tensor-based Markov Decision Process Solver
This implementation relies on modifying the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit, which uses a tabular method to represent the transition models of MDPs, to a tensor-based Value Iteration (see [mdp.py](pymdptoolbox/src/mdptoolbox/mdp.py)). I used the CANDECOMP/PARAFAC decomposition idea to build the transition models as tensor components.

Also, this code generalizes the solution for *n*-dimensional grids.

## Runtimes

Here I show the runtime comparisons between my method (tensor-based computation) and the tabular method used in pymdptoolbox.

| Grid size | Proposed Method (seconds)| Tabular Method (seconds) |
| --- | --- | --- |
| 3x4	 | 0.035378909 | 	0.005196762 |
| 10x10	 | 0.082054114 | 	0.074456691	 |
| 20x20	 | 0.320015955	 | 0.827324724 |
| 30x30	 | 0.971441817	 | 4.431756878 |
| 40x40	 | 2.504667807	 | 14.46240156 |
| 50x50 |  5.316849518	 | 32.75638905 |
| 60x60	 | 9.301800823	 | 67.54549270 |
| 70x70	 | 22.29321971	 | 124.6484179 |
| 80x80	 | 35.35359769	 | 212.2755362 |
| 90x90	 | 48.64809175	 | 340.8302681 |
| 100x100 | 	65.89430323	 | 550.7267468 |
| 150x150 | 	580.3809519	 | 3004.681708 |
| 200x200 | 	855.3208709	 | 8048.422544  |
| 300x300 | 	2899.553054 | 	memory error |
| 400x400 | 	8420.301807	 | memory error |
| 500x500 | 	17139.60858	 | memory error |
| 1000x1000 | 	203646.5906	 | memory error |

For a grid with 200x200 states, our method provides a **89,37%** runtime improvement compared to the tabular method.

## Examples

In file [gridworld.py](gridworld.py) you can set:

### Two-dimension scenario:
```
shape = [3, 4]  # grid size
number_of_obstacles = 1
number_of_terminals = 2
rewards = [100, -100]  # rewards in terminal states
reward_non_terminal_states = -3  # reward in non terminal states, each reward corresponds to a terminal
p_intended = 0.8  # probability of the desired action (intended direction) taking place
```

### Three-dimension scenario:
```
shape = [2, 3, 4]  # grid size
number_of_obstacles = 2
number_of_terminals = 3
rewards = [100, -100, -100]  # rewards in terminal states
reward_non_terminal_states = -3  # reward in non terminal states
p_intended = 0.7  # probability of the desired action (intended direction) taking place
```

### Four-dimension scenario:
```
shape = [2, 3, 8, 10]  # grid size
number_of_obstacles = 2
number_of_terminals = 5
rewards = [100, -100, -100, -100, 100]  # rewards in terminal states, each reward corresponds to a terminal
reward_non_terminal_states = -3  # reward in non terminal states
p_intended = 0.6  # probability of the desired action (intended direction) taking place
```

You also can set manually the positions of obstacles and terminals, but if you prefer, just set the number of obstacles and terminals desired in variables ``number_of_obstacles`` and ``number_of_terminals`` and the code randomly chooses for you.


To the first three dimensions, the actions names are: ``'N', 'S', 'W', 'E', 'B', 'F'``, which corresponds to "North", "South", "West", "East", "Backward" and "Forward". Above three, the names are randomly chosen.