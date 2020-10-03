# A Tensor-based Markov Decision Process Solver
This implementation relies on modifying the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit, which uses a tabular method to represent the transition models of MDPs, to a tensor-based Value Iteration. I used the CANDECOMP/PARAFAC decomposition idea to build the transition models as tensor components.

## Runtimes

Here I show the runtime comparisons between my method (tensor-based computation) and the tabular method used in pymdptoolbox.

| Grid size | Proposed Method (seconds)| Tabular Method (seconds) |
| --- | --- | --- |
| 3x4	 | 0.03537890911 | 	0.005196762085 |
| 10x10	 | 0.08205411434 | 	0.07445669174 |
| 20x20	 | 0.320015955	 | 0.8273247242 |
| 30x30	 | 0.9714418173	 | 4.431756878 |
| 40x40	 | 2.504667807	 | 14.46240156 |
| 50x50 | 5.316849518	 | 32.75638905 |
| 60x60	 | 9.301800823	 | 67.5454927 |
| 70x70	 | 22.29321971	 | 124.6484179 |
| 80x80	 | 35.35359769	 | 212.2755362 |
| 90x90	 | 48.64809175	 | 340.8302681 |
| 100x100 | 	65.89430323	 | 550.7267468 |
| 150x150 | 	580.3809519	 | 3004,681708 |
| 200x200 | 	855.3208709	 | 8048,422544  |
| 300x300 | 	2899.553054 | 	memory error |
| 400x400 | 	8420.301807	 | memory error |
| 500x500 | 	17139.6085838	 | memory error |
| 1000x1000 | 	203646.5906	 | memory error |

For a grid with 200x200 states, our method provides a 89,37% runtime improvement compared to the tabular method.