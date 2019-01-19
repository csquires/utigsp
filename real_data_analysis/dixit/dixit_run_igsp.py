import argparse
import os
import sys
sys.path.append('../..')
from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test, hsic_test
from real_data_analysis.dixit.dixit_meta import get_sample_dict, ESTIMATED_FOLDER, nnodes
import numpy as np

# === PARSE
parser = argparse.ArgumentParser
parser.add_argument('--alpha', type=float)
parser.add_argument('--excluded', type=int)
parser.add_argument('--ci_test', type=str)
args = parser.parse_args()
alpha = args.alpha
excluded = args.excluded
ci_test = args.ci_test

# === LOAD SAMPLES AND REMOVE EXCLUDED
sample_dict, suffstat = get_sample_dict()
sample_dict_exclude = {k: v for k, v in sample_dict.items() if k != frozenset({excluded})}

# === CREATE FILENAME
alpha_invariance = 1e-5
iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % excluded)
filename = os.path.join(iv_estimated_folder,
                        'igsp_%s_alpha=%.2e,alpha_i=%.2e.txt' % (ci_test, alpha, alpha_invariance))

# === RUN ALGORITHM
if not os.path.exists(filename):
    if ci_test == 'gauss_ci':
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

