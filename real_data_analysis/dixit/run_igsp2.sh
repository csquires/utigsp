#!/usr/bin/env bash

for ci_test in gauss_ci hsic
do
    for alpha in .1 .2 .3 .4 .5 .6 .7
    do
        file='test_file'
        echo "python3 -m dixit_run_igsp2.py --alpha ${alpha} --ci_test ${ci_test}" > tmp.sh
        cat slurm_template.sh tmp.sh > ${file}.sh
        rm tmp.sh
        sbatch ${file}.sh
    done
done