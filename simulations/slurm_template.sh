#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 06:00:00
#SBATCH -p newnodes
#SBATCH -J igsp

module add engaging/python/3.6.0
source ../venv/bin/activate
