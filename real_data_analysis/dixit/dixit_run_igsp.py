import argparse
import os
import sys

sys.path.append('../..')
from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test, hsic_test
from real_data_analysis.dixit.dixit_meta import get_sample_dict, ESTIMATED_FOLDER, nnodes
import numpy as np
import multiprocessing

if __name__ == '__main__':
    # === PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--ci_test', type=str)
    args = parser.parse_args()
    alpha = args.alpha
    ci_test = args.ci_test

    # === LOAD SAMPLES AND REMOVE EXCLUDED
    sample_dict, suffstat = get_sample_dict()
    print([v.shape[0] for k, v in sample_dict.items()])
    print(sum(v.shape[0] for k, v in sample_dict.items()))
    obs_samples = sample_dict[frozenset()]
    alpha_invariance = 1e-5

    # === RUN ALGORITHM
    def _run_alg(excluded):
        sample_dict_exclude = {k: v for k, v in sample_dict.items() if k != frozenset({excluded})}
        iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % excluded)
        filename = os.path.join(iv_estimated_folder,
                                'igsp_effective_%s_alpha=%.2e,alpha_i=%.2e.txt' % (ci_test, alpha, alpha_invariance))
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
            elif ci_test == 'hsic':
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
            else:
                raise ValueError
            np.savetxt(filename, est_dag.to_amat())


    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        nodes = list(range(24))
        list(pool.imap_unordered(_run_alg, nodes))



