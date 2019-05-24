import os
import itertools as itr
import sys
sys.path.append('../..')
from simulations.fig1.fig1_settings import *
from config import SIMULATIONS_FOLDER

alphas = [.01]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), alpha in itr.product(
        nsamples_list, nsettings_list, ntargets_list, alphas
    ):
        print(nsamples)
        print(nsettings)
        print(num_unknown)
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--alpha {alpha}'
        os.system(f'cd {SIMULATIONS_FOLDER} && python3 run_icp.py {dag_setting_str} {sample_setting_str} {alg_setting_str}')
        print('done')
