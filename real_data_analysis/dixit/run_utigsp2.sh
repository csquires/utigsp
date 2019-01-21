#!/usr/bin/env bash

for ci_test in gauss_ci
do
    for alpha in .2 .3 .4 .5 .6 .7
    do
        file='test_file'
        echo "python3 dixit_run_utigsp2.py --alpha ${alpha} --ci_test ${ci_test}" > tmp.sh
        cat slurm_template.sh tmp.sh > ${file}.sh
        rm tmp.sh
        sbatch ${file}.sh
    done
done