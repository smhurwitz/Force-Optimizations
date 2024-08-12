# This file runs the fast particle tracing code SIMPLE in embarassingly parallel
# on a Slurm system.

import os
import subprocess

BUILD_DIR = "/global/homes/s/shurwitz/SIMPLE/build/" #SIMPLE build directory
PARETO_DIR = "/global/homes/s/shurwitz/Force-Optimizations/output/QA/with-force-penalty/4/pareto/" #directory for pareto fronts to analyze
CODE_DIR = "/global/homes/s/shurwitz/Force-Optimizations/" #directory for analysis_tools.py

UUIDs = [f.name for f in os.scandir(PARETO_DIR) if f.is_dir()]
for UUID in UUIDs:
    RUN_DIR = PARETO_DIR + UUID + "/"

    python_contents = (f"import sys\n"
                       f"sys.path.append('{CODE_DIR}')\n"
                       f"from analysis_tools import run_SIMPLE\n"
                       f"run_SIMPLE('{UUID}', BUILD_DIR='{BUILD_DIR}')")

    with open(RUN_DIR + "simple.py", "w") as file:
        file.write(python_contents)

    slurm_contents = ("#!/bin/bash\n"
                    "#SBATCH --qos=regular\n"
                    "#SBATCH --time=1440\n"
                    "#SBATCH --constraint cpu\n"
                    "#SBATCH --nodes=1\n"
                    "#SBATCH --ntasks-per-node=1\n"
                    "#SBATCH --ntasks-per-core=2\n"
                    "#SBATCH --cpus-per-task=144\n"
                    "#SBATCH --mail-type=END\n"
                    "#SBATCH --mail-user=shurwitz@umd.edu\n"
                   f'#SBATCH --job-name="SIMPLE_{UUID}"\n'
                    "srun python simple.py")

    with open(RUN_DIR + "simple.sh", "w") as file:
        file.write(slurm_contents)

    command = f"cd {RUN_DIR} && sbatch simple.sh"
    subprocess.run(command, shell=True) 