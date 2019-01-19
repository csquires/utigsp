#!/usr/bin/env bash

for exclude in $(seq 0 23)
do
    for lam in 0.1 1 5 10 50 100 500 1000
    do
        python3 -m dixit_run_gies2.py --lam ${lam} --exclude ${exclude}
    done
done
