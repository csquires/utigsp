#!/usr/bin/env bash

for alpha in 0.0001 0.001 0.01 0.1
do
    file='test_file'
    echo ${file}
    echo "python3 -m test_slurm.py --test ${alpha}" > tmp.sh
    cat slurm_template.sh tmp.sh > ${file}.sh
    rm tmp.sh
    sbatch ${file}.sh
done