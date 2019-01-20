import argparse
import os
import sys
sys.path.append('../..')
from causaldag.inference.structural import unknown_target_igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test, hsic_test
from real_data_analysis.dixit.dixit_meta import ESTIMATED_FOLDER, nnodes, get_sample_dict2
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
alpha_invariance = 1e-5


# === RUN ALGORITHM
def _run_alg(excluded):
    setting_list_exclude = [setting for setting in setting_list if excluded not in setting['known_interventions']]
    iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % excluded)
    filename = os.path.join(iv_estimated_folder,
                            'utigsp_%s_alpha=%.2e,alpha_i=%.2e.txt' % (ci_test, alpha, alpha_invariance))
    os.system('touch %d.tst' % excluded)
    if not os.path.exists(filename):
        if ci_test == 'gauss_ci':
            est_dag = unknown_target_igsp(
                obs_samples,
                setting_list_exclude,
                suffstat,
                nnodes,
                gauss_ci_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance,
                verbose=True
            )
        else:
            est_dag = unknown_target_igsp(
                obs_samples,
                setting_list_exclude,
                obs_samples,
                nnodes,
                hsic_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance
            )
        np.savetxt(filename, est_dag.to_amat())


with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    nodes = list(range(24))
    list(pool.imap_unordered(_run_alg, nodes))

