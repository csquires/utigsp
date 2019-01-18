import numpy as np
import pandas as pd
from config import PROJECT_FOLDER
import os
from causaldag.inference.structural import unknown_target_igsp, igsp
from R_algs.wrappers import run_gies, run_icp
from causaldag.utils.ci_tests import hsic_invariance_test, gauss_ci_test, hsic_test
from real_data_analysis.sachs.sachs_meta import nnodes, SACHS_DATA_FOLDER, ESTIMATED_FOLDER
import shutil
from tqdm import tqdm
os.makedirs(ESTIMATED_FOLDER, exist_ok=True)

# === LOAD SAMPLES
sample_dict = dict()
for file in os.listdir(SACHS_DATA_FOLDER):
    samples = pd.read_csv(os.path.join(SACHS_DATA_FOLDER, file), sep=',')
    iv_str = file.split('=')[1][:-4]
    ivs = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
    sample_dict[ivs] = samples.values
obs_samples = sample_dict[frozenset()]
all_samples = np.concatenate(tuple(sample_dict.values()), axis=0)
suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=obs_samples.shape[0])
suffstat_all = dict(C=np.corrcoef(all_samples, rowvar=False), n=all_samples.shape[0])

# === RUN UNKNOWN TARGET IGSP WITH GAUSS CI
# for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]:
#     filename = os.path.join(ESTIMATED_FOLDER, 'utigsp_gauss_ci_alpha=%.2e.txt' % alpha)
#     if not os.path.exists(filename):
#         est_dag = unknown_target_igsp(
#             sample_dict,
#             suffstat,
#             nnodes,
#             gauss_ci_test,
#             hsic_invariance_test,
#             alpha=alpha,
#             nruns=10,
#             alpha_invariance=alpha
#         )
#         np.savetxt(filename, est_dag.to_amat())

# === RUN UNKNOWN TARGET IGSP WITH HSIC
# for alpha in tqdm([1e-1, 5e-1]):
#     alpha_invariance = 1e-5
#     filename = os.path.join(ESTIMATED_FOLDER, 'utigsp_hsic_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
#     if not os.path.exists(filename):
#         est_dag = unknown_target_igsp(
#             sample_dict,
#             sample_dict[frozenset()],
#             nnodes,
#             hsic_test,
#             hsic_invariance_test,
#             alpha=alpha,
#             nruns=10,
#             alpha_invariance=alpha_invariance,
#             verbose=True
#         )
#         np.savetxt(filename, est_dag.to_amat())

# === RUN IGSP WITH GAUSS CI
# for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
#     alpha_invariance = 1e-5
#     filename = os.path.join(ESTIMATED_FOLDER, 'igsp_gauss_ci_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
#     if not os.path.exists(filename):
#         est_dag = igsp(
#             sample_dict,
#             suffstat,
#             nnodes,
#             gauss_ci_test,
#             hsic_invariance_test,
#             alpha=alpha,
#             nruns=10,
#             alpha_invariance=alpha_invariance
#         )
#         np.savetxt(filename, est_dag.to_amat())

# === RUN IGSP WITH HSIC
# for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
#     alpha_invariance = 1e-5
#     filename = os.path.join(ESTIMATED_FOLDER, 'igsp_hsic_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
#     if not os.path.exists(filename):
#         est_dag = igsp(
#             sample_dict,
#             sample_dict[frozenset()],
#             nnodes,
#             hsic_test,
#             hsic_invariance_test,
#             alpha=alpha,
#             nruns=10,
#             alpha_invariance=alpha_invariance
#         )
#         np.savetxt(filename, est_dag.to_amat())

# === SAVE DATA FOR GIES
sample_folder = os.path.join(PROJECT_FOLDER, 'tmp_sachs')
iv_sample_folder = os.path.join(sample_folder, 'interventional')
os.makedirs(iv_sample_folder, exist_ok=True)
np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
for iv_nodes, samples in sample_dict.items():
    if iv_nodes != frozenset():
        iv_str = 'known_ivs=%s;unknown_ivs=.txt' % ','.join(map(str, iv_nodes))
        np.savetxt(os.path.join(iv_sample_folder, iv_str), samples)

# === RUN ICP
for alpha in tqdm([1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]):
    filename = os.path.join(ESTIMATED_FOLDER, 'icp_alpha=%.2e' % alpha)
    if not os.path.exists(filename):
        amat = run_icp(
            sample_folder,
            alpha
        )
        np.savetxt(filename, amat)

# # === RUN GIES
# est_dags_gies = []
# for lambda_ in [500, 2000]:
#     filename = os.path.join(ESTIMATED_FOLDER, 'gies_lambda=%.2e.txt' % lambda_)
#     if not os.path.exists(filename):
#         amat = run_gies(sample_folder, lambda_)
#         np.savetxt(filename, amat)
#
# shutil.rmtree(sample_folder)