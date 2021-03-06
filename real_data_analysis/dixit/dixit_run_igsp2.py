import argparse
import os
import sys
sys.path.append('../..')
from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test, hsic_test
from real_data_analysis.dixit.dixit_meta import get_sample_dict2, ESTIMATED_FOLDER, nnodes
import numpy as np
import multiprocessing

# === PARSE
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float)
parser.add_argument('--ci_test', type=str)
args = parser.parse_args()
alpha = args.alpha
ci_test = args.ci_test

# === LOAD SAMPLES AND REMOVE EXCLUDED
obs_samples, setting_list = get_sample_dict2()
suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=obs_samples.shape[0])

# === CREATE FILENAME
alpha_invariance = 1e-5


# === RUN ALGORITHM
def _run_alg(excluded):
    # === REMOVE EXCLUDED FROM SAMPLES
    setting_list_exclude = [setting for setting in setting_list if excluded not in setting['known_interventions']]
    sample_dict_exclude = dict()
    for setting in setting_list_exclude:
        iv_nodes = frozenset(setting['known_interventions'])
        samples = setting['samples']
        if iv_nodes not in sample_dict_exclude:
            sample_dict_exclude[iv_nodes] = samples
        else:
            sample_dict_exclude[iv_nodes] = np.concatenate((sample_dict_exclude[iv_nodes], samples), axis=0)

    # === CREATE FILENAME
    iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % excluded)
    filename = os.path.join(iv_estimated_folder,
                            'igsp_%s_alpha=%.2e,alpha_i=%.2e.txt' % (ci_test, alpha, alpha_invariance))
    if not os.path.exists(filename):
        if ci_test == 'gauss_ci':
            os.system('touch gauss_ci_igsp_%d.tst' % excluded)
            est_dag = igsp(
                sample_dict_exclude,
                suffstat,
                nnodes,
                gauss_ci_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance
            )
        else:
            os.system('touch hsic_igsp_%d.tst' % excluded)
            est_dag = igsp(
                sample_dict_exclude,
                sample_dict_exclude[frozenset()],
                nnodes,
                hsic_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance
            )
        np.savetxt(filename, est_dag.to_amat())


with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    pool.imap_unordered(_run_alg, list(range(24)))
