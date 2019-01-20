#!/usr/bin/env bash

for exclude in $(seq 0 23)
do
    for lam in 0 0.001
    do
        python3 -m dixit_run_gies2.py --lam ${lam} --exclude ${exclude}
    done
done
