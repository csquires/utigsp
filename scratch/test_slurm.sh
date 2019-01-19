#!/bin/bash

#SBATCH -N 32
#SBATCH -n 32
#SBATCH --time 0:00:10
#SBATCH -p sched_mit_nse

module add engaging/python/3.6.0
source venv/bin/activate

python3 -m scratch/test_slurm.py --test hi

