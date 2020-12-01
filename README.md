# A Tensor-based Markov Decision Process Solver (CP-MDP)

**CP-MDP** is an implementation based on the MICAI 2020 paper: 

> Daniela Kuinchtner, Felipe Meneguzzi and Afonso Sales.<br>
> **[A Tensor-Based Markov Decision Process Representation](https://doi.org/10.1007/978-3-030-60884-2_23)**<br>
> In Advances in *Soft Computing* (pp. 313-324).

This implementation relies on modifying the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit, which uses a tabular method to represent the transition models of MDPs and a tabular value iteration to compute the utilities, to a tensor decomposition value iteration (see [mdp.py](src/tensor_method_ndimensional/pymdptoolbox/src/mdptoolbox/mdp.py)). I use the CANDECOMP/PARAFAC decomposition idea (which gives the name CP-MDP for our solver) to build the transition models as tensor components in a compact representation.

Also, this code generalizes the solution for *n*-dimensional grids.
  

## Examples

In file [gridworld.py](src/tensor_method_ndimensional/gridworld.py) you can set:

### Two-dimension scenario:
```
shape = [3, 4]  # grid size
number_of_obstacles = 1
number_of_terminals = 2
rewards = [100, -100]  # rewards in terminal states, each reward corresponds to a terminal
reward_non_terminal_states = -3  # reward in non terminal states
p_intended = 0.8  # probability of the desired action (intended direction) taking place
```

### Three-dimension scenario:
```
shape = [2, 3, 4]  # grid size
number_of_obstacles = 2
number_of_terminals = 3
rewards = [100, -100, -100]  # rewards in terminal states, each reward corresponds to a terminal
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
