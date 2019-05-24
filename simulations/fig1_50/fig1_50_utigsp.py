import os
import itertools as itr
import sys
sys.path.append('../..')
from simulations.fig1_50.fig1_50_settings import *
from config import SIMULATIONS_FOLDER

alpha_list = [1e-5]
alpha_invariant_list = [1e-5]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), alpha, alpha_invariant in itr.product(
        nsamples_list, nsettings_list, ntargets_list, alpha_list, alpha_invariant_list
    ):
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--alpha {alpha} --alpha_invariant {alpha_invariant}'
        full_command = f'python3 run_utigsp.py {dag_setting_str} {sample_setting_str} {alg_setting_str}'
        if SERVER:
            os.system(f'cd {SIMULATIONS_FOLDER} && echo "{full_command}" > tmp.sh')
            os.system(f'cd {SIMULATIONS_FOLDER} && cat slurm_template.sh tmp.sh > job.sh')
            os.system(f'cd {SIMULATIONS_FOLDER} && rm tmp.sh')
            os.system(f'cd {SIMULATIONS_FOLDER} && sbatch job.sh')
        else:
            os.system(f'cd {SIMULATIONS_FOLDER} && {full_command}')
