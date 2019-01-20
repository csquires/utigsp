#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 6:00:00
#SBATCH -p newnodes
#SBATCH -J dixit-analysis

module add engaging/python/3.6.0
source ../venv/bin/activate
python3 run_utigsp.py --nnodes 20 --nneighbors 1.5 --ndags 50 --nsamples 500 --nsettings 20 --num_known 1 --num_unknown 8 --intervention perfect1 --alpha 1e-05 --alpha_invariant 1e-05
