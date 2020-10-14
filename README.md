# A Tensor-based Markov Decision Process Solver (CP-MDP)

**CP-MDP** is an implementation based on the MICAI 2020 paper: 

> Daniela Kuinchtner, Felipe Meneguzzi and Afonso Sales.<br>
> **[A Tensor-Based Markov Decision Process Representation](https://doi.org/10.1007/978-3-030-60884-2_23)**<br>
> In Advances in *Soft Computing* (pp. 313-324).

This implementation relies on modifying the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit, which uses a tabular method to represent the transition models of MDPs, to a tensor-based Value Iteration (see [mdp.py](pymdptoolbox/src/mdptoolbox/mdp.py)). I use the CANDECOMP/PARAFAC decomposition idea (which gives the name CP-MDP for our solver) to build the transition models as tensor components in a compact representation.

Also, this code generalizes the solution for *n*-dimensional grids.

## Runtimes

Here I show runtime comparisons between my method (tensor-based computation) and the tabular method used in pymdptoolbox.

### Runtime (seconds)


| Dimensions|Grid size | Proposed Method| Tabular Method  | Improvement|
|---| --- | --- | --- |---|
|2D| 3x4	 | 0.038637638| 	0.008386611|-360.71%|
|---| 20x20	 | 0.909860611	|0.357823848	|-154.28%|
|---| 50x50 |  4.266415358	|13.58046985	|68.58%|
| ---|80x80	 |17.11296272	|88.35849571	|80.63%|
|---| 100x100 | 35.59322596|	184.0137181	|80.66%|
|---| 150x150 | 	64.00285721|	196.4976344	|67.43%|
|---| 300x300 | 	648.1915321| 	unable to allocate ||
|---| 500x500 | 	2732.505154	 | unable to allocate ||
|--- |1000x1000 | 	25240.89330	 | unable to allocate ||
|3D|2x3x4|0.044556617	|0.033081769	|-34.69%|
|---|10x10x10|1.940085411|	2.358873367|	17,75%|
|---|20x20x20|19.93034005	|49.11039615	|59.42%|
|---|20x30x40|63.30102301	|134.2824292	|52.86%|
|---|50x50x50|493.3360932| unable to allocate ||
|4D|2x3x4x5|0.357324123|	0.349513769	|-2.23%|
|---|10x10x10x10|35.40316582|	61.91894412	|42.82%|
|---|15x10x15x10|87.53485560|	151.9464970|	42.39%|
|---|20x30x40x50|9054.412815|unable to allocate ||
|5D| 2x3x4x5x6|3.563772917	|4.437751373	|19.69%|
|---|5x5x5x5x5|14.28819275	|16.86444163	|15.28%|
|---|8x8x8x8x8|162.1390219|	321.5184011	|49.57%|
|---|10x10x10x10x10|527.1325731|unable to allocate ||
|6D|2x3x4x5x6x7|39.43572807	|60.12430819	|34.41%|
|---|5x5x5x5x5x5|105.3635645	|164.7636745|	36.05%|
|---|6x6x6x5x5x5|183.0221536	|322.2038546	|43.20%|
|---|10x10x10x10x10x10|8287.202552|unable to allocate ||

### Memory used (Mb)


| Dimensions|Grid size | Proposed Method| Tabular Method  | Improvement|
|---| --- | --- | --- |---|
|2D| 3x4	 | 69.320704	|69.599232|	0.40%|
|---| 20x20	 | 70.176768|	74.616832|	5.95%|
|---| 50x50 | 73.986048	|119.48851	|38.08%|
| ---|80x80	 |84.377600	|215.85510	|60.91%|
|---| 100x100 | 91.357184	|315.05203	|71.00%|
|---| 150x150 | 	113.97529|	681.96761|	83.29%|
|---| 300x300 | 	479.92012| 	unable to allocate ||
|---| 500x500 | 	1691.4513	 | unable to allocate ||
|--- |1000x1000 | 	13099.646	 | unable to allocate ||
|3D|2x3x4|69.320704|	69.214208	|-0.15%|
|---|10x10x10|71.946240	|102.52288|	29.82%|
|---|20x20x20|86.847488|	562.60608|84.56%|
|---|20x30x40|126.75891	|1887.9078	|93.29%|
|---|50x50x50|401.12537| unable to allocate ||
|4D|2x3x4x5|69.816320|	69.685248	|-0.19%|
|---|10x10x10x10|99.397632	|1109.1558	|91.04%|
|---|15x10x15x10|137.97376	|2581.5982	|94.66%|
|---|20x30x40x50|4335.7143|unable to allocate ||
|5D| 2x3x4x5x6|72.994816	|110.41177	|33.89%|
|---|5x5x5x5x5|82.169856	|449.61382	|81.72%|
|---|8x8x8x8x8|196.74726|	6128.8161|96.79%|
|---|10x10x10x10x10|456.94156|unable to allocate ||
|6D|2x3x4x5x6x7|95.277056	|10550039|90.97%|
|---|5x5x5x5x5x5|147.66899|	35907461	|95.89%|
|---|6x6x6x5x5x5|203.27219|	6389.1660|96.82%|
|---|10x10x10x10x10x10|5135.2780|unable to allocate ||


## Examples

In file [gridworld.py](gridworld.py) you can set:

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
