#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 00:10:00
#SBATCH -p newnodes
#SBATCH -J sched_engaging_default

module add engaging/python/3.6.0
source ../venv/bin/activate
