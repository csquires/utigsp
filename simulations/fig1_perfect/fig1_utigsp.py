import os
import itertools as itr
import sys
sys.path.append('../..')
from simulations.fig1_perfect.fig1_settings import *
from config import SIMULATIONS_FOLDER

alpha_list = [1e-5]
alpha_invariant_list = [1e-5]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), alpha, alpha_invariant in itr.product(
        nsamples_list, nsettings_list, ntargets_list, alpha_list, alpha_invariant_list
    ):
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags} --nonlinear {nonlinear}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--alpha {alpha} --alpha_invariant {alpha_invariant}'
        full_command = f'python3 run_utigsp.py {dag_setting_str} {sample_setting_str} {alg_setting_str}'
        # os.system(f'cd .. && echo "{full_command}" > tmp.sh')
        # os.system('cd .. && cat slurm_template.sh tmp.sh > job.sh')
        # os.system('cd .. && rm tmp.sh')
        # os.system('cd .. && sbatch job.sh')
        os.system(f'cd {SIMULATIONS_FOLDER} && {full_command}')
