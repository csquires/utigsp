#!/usr/bin/env bash

for ci_test in gauss_ci hsic
do
    for alpha in .001 .005 .01 .05 .1 .2 .3 .4 .5
    do
        file='test_file'
        echo "python3 dixit_run_utigsp.py --alpha ${alpha} --ci_test ${ci_test}" > tmp.sh
        cat slurm_template.sh tmp.sh > ${file}.sh
        rm tmp.sh
        sbatch ${file}.sh
    done
done