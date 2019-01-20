import argparse
import os
import numpy as np
import causaldag as cd
from causaldag.inference.structural import unknown_target_igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
import multiprocessing

import sys
sys.path.append('..')
from simulations.create_dags_and_samples import save_dags_and_samples, get_dag_samples, get_sample_folder

# === DEFINE PARSE
parser = argparse.ArgumentParser()
parser.add_argument('--nnodes', type=int)
parser.add_argument('--ndags', type=int)
parser.add_argument('--nneighbors', type=float)

parser.add_argument('--nsamples', type=int)
parser.add_argument('--nsettings', type=int)
parser.add_argument('--num_known', type=int)
parser.add_argument('--num_unknown', type=int)
parser.add_argument('--intervention', type=str)

parser.add_argument('--nruns', type=int)
parser.add_argument('--depth', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--alpha_invariant', type=float)

# === PARSE ARGUMENTS
args = parser.parse_args()

ndags = args.ndags
nnodes = args.nnodes
nneighbors = args.nneighbors

nsamples = args.nsamples
nsettings = args.nsettings
num_known = args.num_known
num_unknown = args.num_unknown
intervention = args.intervention

nruns = args.nruns
depth = args.depth
alpha = args.alpha
alpha_invariant = args.alpha_invariant


# === CREATE DAGS AND SAMPLES: THIS MUST BE DONE THE SAME WAY EVERY TIME FOR THIS TO WORK
save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention)


def _run_utigsp(dag_num):
    # === GENERATE FILENAME
    sample_folder = get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)
    alg_folder = os.path.join(sample_folder, 'estimates', 'utigsp')
    os.makedirs(alg_folder, exist_ok=True)
    filename = os.path.join(alg_folder, 'nruns=%d,depth=%d,alpha=%.2e,alpha_invariant=%.2e' % (nruns, depth, alpha, alpha_invariant))

    # === RUN ALGORITHM
    if not os.path.exists(filename):
        obs_samples, setting_list, _ = get_dag_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)
        suffstat = np.corrcoef(obs_samples, rowvar=False)
        est_dag = unknown_target_igsp(
            obs_samples,
            setting_list,
            suffstat,
            nnodes,
            gauss_ci_test,
            hsic_invariance_test,
            alpha=alpha,
            alpha_invariance=alpha_invariant,
            depth=depth,
            nruns=nruns
        )

        np.savetxt(filename, est_dag.to_amat())


with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    pool.map(_run_utigsp, list(range(ndags)))
