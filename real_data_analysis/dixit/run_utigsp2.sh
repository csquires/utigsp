#!/usr/bin/env bash

for ci_test in gauss_ci hsic
do
    for alpha in 0.0001 0.001 0.01 0.1
    do
        file='test_file'
        echo "python3 -m dixit_run_utigsp2.py --alpha ${alpha} --ci_test ${ci_test}" > tmp.sh
        cat slurm_template.sh tmp.sh > ${file}.sh
        rm tmp.sh
        sbatch ${file}.sh
    done
done