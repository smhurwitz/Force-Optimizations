import sys
# sys.path.append("/global/homes/s/shurwitz/Force-Optimizations/" )

from optimization_tools import *

iteration = int(sys.argv[1])
INPUT_DIR=f"../output/QA/without-force-penalty/{iteration-1}/pareto/"
OUTPUT_DIR=f"../output/QA/without-force-penalty/{iteration}/optimizations/"
for i in range(250): continuation(N=1, dx=10**rand(-3, -1.3), INPUT_DIR=INPUT_DIR, OUTPUT_DIR=OUTPUT_DIR)