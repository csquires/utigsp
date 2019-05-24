import os
import itertools as itr
import sys
sys.path.append('../..')
from simulations.fig1.fig1_settings import *
from config import SIMULATIONS_FOLDER

lams = [1]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), lam in itr.product(
        nsamples_list, nsettings_list, ntargets_list, lams
    ):
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--lam {lam}'
        os.system(f'cd {SIMULATIONS_FOLDER} && python3 run_gies.py {dag_setting_str} {sample_setting_str} {alg_setting_str}')
