import pandas as pd
import os
from config import PROJECT_FOLDER
from real_data_analysis.dixit.dixit_meta import DIXIT_DATA_FOLDER, ESTIMATED_FOLDER, nnodes
from causaldag.inference.structural import igsp, unknown_target_igsp
from causaldag.utils.ci_tests import hsic_test, gauss_ci_test, hsic_invariance_test
import numpy as np
from tqdm import tqdm
from R_algs.wrappers import run_gies
import multiprocessing

sample_dict = {}
ivs = []
for file in os.listdir(DIXIT_DATA_FOLDER):
    samples = pd.read_csv(os.path.join(DIXIT_DATA_FOLDER, file), sep=',')
    iv_str = file.split('=')[1][:-4]
    iv = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
    sample_dict[iv] = samples.values
    if iv_str != '': ivs.append(int(iv_str))


obs_samples = sample_dict[frozenset()]
suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=obs_samples.shape[0])


def run_algs(iv):
    iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % iv)
    os.makedirs(iv_estimated_folder, exist_ok=True)
    sample_dict_exclude = {k: v for k, v in sample_dict.items() if k != frozenset({iv})}

    # # === RUN IGSP WITH HSIC
    # for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
    #     alpha_invariance = 1e-5
    #     filename = os.path.join(iv_estimated_folder,
    #                             'igsp_hsic_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
    #     if not os.path.exists(filename):
    #         est_dag = igsp(
    #             sample_dict_exclude,
    #             sample_dict[frozenset()],
    #             nnodes,
    #             hsic_test,
    #             hsic_invariance_test,
    #             alpha=alpha,
    #             nruns=10,
    #             alpha_invariance=alpha_invariance,
    #         )
    #         np.savetxt(filename, est_dag.to_amat())

    # === RUN IGSP WITH GAUSS CI
    for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
        alpha_invariance = 1e-5
        filename = os.path.join(iv_estimated_folder,
                                'igsp_gauss_ci_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
        if not os.path.exists(filename):
            est_dag = igsp(
                sample_dict_exclude,
                suffstat,
                nnodes,
                gauss_ci_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance,
            )
            np.savetxt(filename, est_dag.to_amat())

    # # === RUN UTIGSP WITH HSIC
    # for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
    #     alpha_invariance = 1e-5
    #     filename = os.path.join(iv_estimated_folder,
    #                             'utigsp_hsic_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
    #     if not os.path.exists(filename):
    #         est_dag = unknown_target_igsp(
    #             sample_dict_exclude,
    #             sample_dict[frozenset()],
    #             nnodes,
    #             hsic_test,
    #             hsic_invariance_test,
    #             alpha=alpha,
    #             nruns=10,
    #             alpha_invariance=alpha_invariance
    #         )
    #         np.savetxt(filename, est_dag.to_amat())

    # === RUN UTIGSP WITH GAUSS CI
    for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
        alpha_invariance = 1e-5
        filename = os.path.join(iv_estimated_folder,
                                'utigsp_gauss_ci_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
        if not os.path.exists(filename):
            est_dag = unknown_target_igsp(
                sample_dict_exclude,
                suffstat,
                nnodes,
                gauss_ci_test,
                hsic_invariance_test,
                alpha=alpha,
                nruns=10,
                alpha_invariance=alpha_invariance
            )
            np.savetxt(filename, est_dag.to_amat())

    # === SAVE DATA FOR GIES
    # sample_folder = os.path.join(PROJECT_FOLDER, 'tmp_dixit_iv=%s' % iv)
    # iv_sample_folder = os.path.join(sample_folder, 'interventional')
    # os.makedirs(iv_sample_folder, exist_ok=True)
    # np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
    # for iv_nodes, samples in sample_dict_exclude.items():
    #     if iv_nodes != frozenset():
    #         iv_str = 'known_ivs=%s;unknown_ivs=.txt' % ','.join(map(str, iv_nodes))
    #         np.savetxt(os.path.join(iv_sample_folder, iv_str), samples)
    #
    # # === RUN GIES
    # for lambda_ in tqdm([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]):
    #     filename = os.path.join(iv_estimated_folder, 'gies_lambda=%.2e.txt' % lambda_)
    #     if not os.path.exists(filename):
    #         amat = run_gies(
    #             sample_folder,
    #             lambda_
    #         )
    #         np.savetxt(filename, amat)


with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
    p.map(run_algs, ivs)

