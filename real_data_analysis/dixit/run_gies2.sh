#!/usr/bin/env bash

for exclude in $(seq 0 23)
do
    for lam in .001 .01 .1 1 5 10 50 100 500
    do
        python3 dixit_run_gies2.py --lam ${lam} --exclude ${exclude}
    done
done
