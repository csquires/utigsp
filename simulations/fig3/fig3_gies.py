import os
import itertools as itr

nnodes = 20
nneighbors = 1.5
ndags = 100
nsamples_list = [100, 300, 500, 1000]
nsettings_list = [3]
ntargets_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
intervention = 'perfect1'
lams = [1]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), lam in itr.product(
        nsamples_list, nsettings_list, ntargets_list, lams
    ):
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--lam {lam}'
        os.system(f'cd .. && python3 run_gies.py {dag_setting_str} {sample_setting_str} {alg_setting_str}')
