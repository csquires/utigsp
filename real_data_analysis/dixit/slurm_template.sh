#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 6:00:00
#SBATCH -p sched_mit_hill
#SBATCH -J test

module add engaging/python/3.6.0
source ../../venv/bin/activate
