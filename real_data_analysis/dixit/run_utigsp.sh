#!/usr/bin/env bash

for ci_test in gauss_ci hsic
do
    for exclude in 2 9 15 16 17 20 21 22
    do
        for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
        do
            file='test_file'
            echo "python3 -m dixit_run_utigsp.py --alpha ${alpha} --exclude ${exclude} --ci_test ${ci_test}" > tmp.sh
            cat slurm_template.sh tmp.sh > ${file}.sh
            rm tmp.sh
            sbatch ${file}.sh
        done
    done
done