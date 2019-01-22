#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 4:00:00
#SBATCH -p sched_engaging_default
#SBATCH -J simulations

module add engaging/python/3.6.0
source ../venv/bin/activate
