#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 12:00:00
#SBATCH -p newnodes
#SBATCH -J dixit-analysis

module add engaging/python/3.6.0
source ../../venv/bin/activate
python3 dixit_run_utigsp2.py --alpha 0.1 --ci_test gauss_ci
