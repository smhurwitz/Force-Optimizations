#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=200
#SBATCH --mail-type=END
#SBATCH --mail-user=shurwitz@umd.edu
#SBATCH --constraint cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2

python gen_pareto.py $1
srun python continuation.py $1