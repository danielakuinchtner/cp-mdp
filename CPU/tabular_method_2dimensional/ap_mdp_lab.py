# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
sys.path.insert(0, 'pymdptoolbox/src')
import mdptoolbox.example
import time
from gen_scenario import *


"""
(Y,X)
| 00 01 02 ... 0X-1
| 10  .         .
| 20    .       .
| .       .     .
| .         .   .
| .           . .
| Y-1,0 . . .   Y-1X-1
"""


shape = [3,4]
rewards = [[0,3,100],[1,3,-100]]
obstacles = [[1,1]]
terminals = [[0,3],[1,3]]

start_time_all = time.time()
start_time_succ = time.time()
P, R = mdp_grid(shape=shape, terminals=terminals, r=-3, rewards=rewards, obstacles=obstacles)
print("\n--- Computed successors and rewards in: %s seconds ---" % (time.time() - start_time_succ))
start_time_vi = time.time()
vi = mdptoolbox.mdp.ValueIterationGS(P, R, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)

# vi.verbose = True # Uncomment this for question 2
vi.run()
print("\n--- Solved with VI in: %s seconds ---" % (time.time() - start_time_vi))
print("\n--- Solved all in: %s seconds ---" % (time.time() - start_time_all))

start_time_pi = time.time()
pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=1, policy0=None, max_iter=1000, eval_type=0, skip_check=True)
pi.run()
print("\n--- Solved with PI in: %s seconds ---" % (time.time() - start_time_pi))

start_time_mpi = time.time()
mpi = mdptoolbox.mdp.PolicyIterationModified(P, R, discount=1, epsilon=0.001, max_iter=1000, skip_check=True)
mpi.run()
print("\n--- Solved with MPI in: %s seconds ---" % (time.time() - start_time_mpi))

print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)
print_policy(pi.policy, shape, obstacles=obstacles, terminals=terminals)
print_policy(mpi.policy, shape, obstacles=obstacles, terminals=terminals)


start_time_lp = time.time()
lp = mdptoolbox.mdp._LP(P, R, discount=1, skip_check=False)
lp.run()
print("\n--- Solved with LP in: %s seconds ---" % (time.time() - start_time_lp))
print_policy(lp.policy, shape, obstacles=obstacles, terminals=terminals)


start_time_fh = time.time()
fh = mdptoolbox.mdp.FiniteHorizon(P, R, discount=1, N=10000, h=None, skip_check=False)
fh.run()
print("\n--- Solved with FH in: %s seconds ---" % (time.time() - start_time_fh))
print_policy(fh.policy, shape, obstacles=obstacles, terminals=terminals)


start_time_ql = time.time()
ql = mdptoolbox.mdp.QLearning(P, R, discount=1, n_iter=10000, skip_check=False)
ql.run()
print("\n--- Solved with QL in: %s seconds ---" % (time.time() - start_time_ql))
print_policy(ql.policy, shape, obstacles=obstacles, terminals=terminals)