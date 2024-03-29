#!/usr/bin/env bash

for ci_test in gauss_ci hsic
do
    for exclude in $(seq 0 23)
    do
        for alpha in 0.0001 0.001 0.01 0.1
        do
            file='test_file'
            echo "python3 -m dixit_run_icp2.py --alpha ${alpha} --exclude ${exclude} --ci_test ${ci_test}" > tmp.sh
            cat slurm_template_R.sh tmp.sh > ${file}.sh
            rm tmp.sh
            sbatch ${file}.sh
        done
    done
done