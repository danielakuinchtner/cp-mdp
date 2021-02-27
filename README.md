# A Tensor-based Markov Decision Process Solver (CP-MDP)

**CP-MDP** is an implementation based on the MICAI 2020 paper: 

> Daniela Kuinchtner, Felipe Meneguzzi and Afonso Sales.<br>
> **[A Tensor-Based Markov Decision Process Representation](https://doi.org/10.1007/978-3-030-60884-2_23)**<br>
> In Advances in *Soft Computing* (pp. 313-324).

This implementation relies on modifying the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit, which uses a tabular method to represent the transition models of MDPs and a tabular value iteration and policy iteration to solve the problem, to a tensor decomposition value iteration and policy iteration compact algorihtms (see [mdp.py](src/cp-mdp/pymdptoolbox/mdp.py)). 

It uses the CANDECOMP/PARAFAC decomposition idea (which gives the name CP-MDP for our solver) to build the transition models as tensor components in a compact representation.

This code solves the GridWorld problem for *n*-dimensional grids.
  


