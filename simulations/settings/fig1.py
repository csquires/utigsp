import os
import itertools as itr

nnodes = 20
nneighbors = 1.5
ndags = 50
nsamples_list = [100, 300, 500]
nsettings_list = [20]
ntargets_list = [(1, 0), (1, 2), (1, 4), (1, 6), (1, 8)]
intervention = 'perfect1'
alpha_list = [1e-5]
alpha_invariant_list = [1e-5]


if __name__ == '__main__':
    for nsamples, nsettings, (num_known, num_unknown), alpha, alpha_invariant in itr.product(
        nsamples_list, nsettings_list, ntargets_list, alpha_list, alpha_invariant_list
    ):
        dag_setting_str = f'--nnodes {nnodes} --nneighbors {nneighbors} --ndags {ndags}'
        sample_setting_str = f'--nsamples {nsamples} --nsettings {nsettings} --num_known {num_known} --num_unknown {num_unknown} --intervention {intervention}'
        alg_setting_str = f'--alpha {alpha} --alpha_invariant {alpha_invariant}'
        full_command = f'cd .. && python3 -m run_utigsp.py {dag_setting_str} {sample_setting_str} {alg_setting_str}'
        os.system(f'cd .. && echo "{full_command}" > tmp.sh')
        os.system('cd .. && cat slurm_template.sh tmp.sh > job.sh')
        os.system('cd .. && rm tmp.sh')
        os.system('cd .. && sbatch job.sh')
