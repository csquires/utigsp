#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 0:01:00
#SBATCH -p sched_mit_hill
#SBATCH -J test

module add engaging/python/3.6.0
R=/home/csquires/R/x86_64-pc-linux-gnu-library/3.4
source ../../venv/bin/activate
