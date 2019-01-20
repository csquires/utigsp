#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 6:00:00
#SBATCH -p newnodes
#SBATCH -J dixit-analysis

module add engaging/python/3.6.0
source ../../venv/bin/activate
cd .. && python3 -m run_utigsp.py --nnodes 100 --nneighbors 1.5 --ndags 50 --nsamples 500 --nsettings 3 --num_known 3 --num_unknown 0 --intervention inhibitory1 --alpha 0.1 --alpha_invariant 1e-05
