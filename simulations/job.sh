#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 2:00:00
#SBATCH -p sched_any
#SBATCH -J simulations

module add engaging/python/3.6.0
source ../venv/bin/activate
python3 run_igsp.py --nnodes 10 --nneighbors 1.5 --ndags 50 --nsamples 500 --nsettings 5 --num_known 1 --num_unknown 3 --intervention perfect1 --alpha 1e-05 --alpha_invariant 1e-05
