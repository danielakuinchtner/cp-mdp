# mdp
This implementation relies on the [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) toolkit.

## MDPs in pymdptoolbox - Class Assignment

MDP planning implemented in a mathematical toolkit.
The following code sets up an MDP environment and computes the policy for the given MDP using the Value Iteration.

<p align="center">
<img src="mdp_simple.png"/>
</p>

## Runtimes

I run all tests using an Intel Core i7-4500U CPU with 8GB RAM and 64-bit Windows operator system.

| Grid size | My approach (seconds) | Tradicional approach (seconds) |
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
| 150x150 | 	627.0185016	 | 20304.10504 |
| 200x200 | 	2196.518497	 | memory error  |
| 300x300 | 	11588.25407 | 	memory error |
| 400x400 | 	didn't test yet	 | memory error |
| 500x500 | 	didn't test yet	 | memory error |
| 1000x1000 | 	didn't test yet	 | memory error |
| 10000x10000	 | didn't test yet	 | memory error |
| 100000x100000	 | didn't test yet	 | memory error |
| 1000000x1000000 | didn't test yet	 | 	memory error |
