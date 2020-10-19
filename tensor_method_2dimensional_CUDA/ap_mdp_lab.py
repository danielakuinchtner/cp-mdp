# The line below is to be used if you have pymdptoolbox installed with setuptools
# import mdptoolbox.example
# Whereas the line below obviate the need to install that
import sys
sys.path.insert(1,'pymdptoolbox/src')
import mdptoolbox.example
import time
from gen_scenario import *
from numba import cuda
from numba import *
"""
(Y,X)
| 00 01 02 ... 0X-1       'N' = North
| 10  .         .         'S' = South
| 20    .       .         'W' = West
| .       .     .         'E' = East
| .         .   .         'T' = Terminal
| .           . .         'O' = Obstacle
| Y-1,0 . . .   Y-1X-1
"""

# install openblas lib to improve even more the runtime: conda install -c anaconda openblas

shape = [3, 4]
rewards = [[0, 3, 100], [1, 3, -100]]
obstacles = [[1, 1]]
terminals = [[0, 3], [1, 3]]
actions = ['N', 'E', 'W', 'S']
final_limits = [shape[0]-1, shape[1]-1]
states = shape[0] * shape[1]
print("Executing a", shape, "grid")

start_time1 = time.time()
blockdim = (32, 8)
griddim = (32,16)
d_shape = cuda.to_device(shape)
succ_xy, origin_xy, probability_xy, R = mdp_grid[griddim, blockdim](shape=d_shape, terminals=terminals, r=-3,
                                                                         rewards=rewards, obstacles=obstacles,
                                                                         actions=actions, final_limits=final_limits)
d_shape.to_host()
print("--- Succ: %s seconds ---" % (time.time() - start_time1))

start_time = time.time()
vi = mdptoolbox.mdp.ValueIterationGS(shape, terminals, obstacles, succ_xy, probability_xy, R, states,
                                     discount=1, epsilon=0.001, max_iter=1000, skip_check=True)

vi.run()

print("--- VI: %s seconds ---" % (time.time() - start_time2))

print_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals, actions=actions)
# display_policy(vi.policy, shape, obstacles=obstacles, terminals=terminals)

# print(output)
# print(R)

# ----------------------------------------------
# RESULTS OF MY ALGORITHM
# ----------------------------------------------

# 3x4
# --- 0.014991998672485352 seconds ---
# --- 0.015987873077392578 seconds ---
# --- 0.013991594314575195 seconds ---
# --------------- 0,0149 average

# 10x10
# --- 0.17789649963378906 seconds ---
# --- 0.21387767791748047 seconds ---
# --- 0.21387767791748047 seconds ---
# --------------- 0,1861 average

# 30x30
# --- 2.0977985858917236 seconds --- diminuiu 50% do tempo

# 40x40
# --- 5.280977487564087 seconds --- diminuiu 61% do tempo

# 50x50
# --- 11.123634099960327 seconds --- diminuiu 65% do tempo
# --- 7.264829158782959 seconds ---

# 100x100
# --- 108.47347044944763 seconds --- 1min diminuiu 82% do tempo

# 160x160
# --- 1083.288479566574 seconds --- 18min diminuiu mais de 93% do tempo

# 200x200
# --- 2216.2952921390533 seconds --- 36min

# 300x300
# vai

# 400x400
# vai

# 500x500
# vai

# 1000x1000
# vai

# 2000x2000
# vai

# 10000x10000
# vai

# 50000x50000
# vai

# 100000x100000
# vai

# 1000000x1000000
# vai


# ----------------------------------------------
# NORMAL VALUE ITERATION RESULTS (OLD ALGORITHM):
# ----------------------------------------------

# 3x4
# --- 0.0059969425201416016 seconds ---
# --- 0.004996299743652344 seconds ---
# --- 0.006995201110839844 seconds ---
# --------------- 0,0059 average

# 10x10
# --- 0.07695412635803223 seconds ---
# --- 0.06796026229858398 seconds ---
# --- 0.07095956802368164 seconds ---
# --------------- 0,0719 average

# 30x30
# --- 4.072670221328735 seconds ---

# 40x40
# --- 13.774938583374023 seconds ---

# 50x50
# --- 31.22613263130188 seconds ---

# 100x100
# --- 590.0848195552826 seconds --- 9min

# 160x160
# more than 4h15min

# 170x170
# memory error

# 200x200
# memory error

# 300x300
# memory error

# 400x400
# memory error

# 500x500
# memory error

# 1000x1000
# memory error
