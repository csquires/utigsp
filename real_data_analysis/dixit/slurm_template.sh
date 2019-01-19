#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 0:00:10
#SBATCH -p newnodes
#SBATCH -J test

module add engaging/python/3.6.0
source ../venv/bin/activate
