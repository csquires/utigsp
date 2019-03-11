#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16 
#SBATCH --time 12:00:00
#SBATCH -p newnodes
#SBATCH -J soft_interventions 

module add engaging/python/3.6.0
source ../venv/bin/activate
