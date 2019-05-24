#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16 
#SBATCH --time 12:00:00
#SBATCH -p newnodes
#SBATCH -J soft_interventions 

module add engaging/python/3.6.0
source ../venv/bin/activate
python3 run_utigsp.py --nnodes 50 --nneighbors 5 --ndags 100 --nsamples 500 --nsettings 3 --num_known 1 --num_unknown 3 --intervention perfect2 --alpha 1e-05 --alpha_invariant 1e-05
