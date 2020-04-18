import numpy as np
import pandas as pd
from config import PROJECT_FOLDER
import os
from causaldag import unknown_target_igsp, igsp
from R_algs.wrappers import run_gies, run_icp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_test, MemoizedCI_Tester, gauss_ci_suffstat
from causaldag.utils.invariance_tests import hsic_invariance_test, MemoizedInvarianceTester, gauss_invariance_test, gauss_invariance_suffstat
from real_data_analysis.sachs.sachs_meta import nnodes, SACHS_DATA_FOLDER, ESTIMATED_FOLDER
import shutil
from tqdm import tqdm
os.makedirs(ESTIMATED_FOLDER, exist_ok=True)
import json

# === LOAD SAMPLES
sample_dict = dict()
for file in os.listdir(SACHS_DATA_FOLDER):
    samples = pd.read_csv(os.path.join(SACHS_DATA_FOLDER, file), sep=',')
    iv_str = file.split('=')[1][:-4]
    ivs = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
    sample_dict[ivs] = samples.values
obs_samples = sample_dict[frozenset()]
all_samples = np.concatenate(tuple(sample_dict.values()), axis=0)
suffstat = gauss_ci_suffstat(obs_samples)
suffstat_all = dict(C=np.corrcoef(all_samples, rowvar=False), n=all_samples.shape[0])

setting_list = [
    {'known_interventions': iv_nodes}
    for iv_nodes, samples in sample_dict.items()
    if iv_nodes != frozenset()
]
iv_samples_list = [sample_dict[setting['known_interventions']] for setting in setting_list]
invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

# === RUN UNKNOWN TARGET IGSP WITH GAUSS CI
for alpha in [6e-1, 7e-1]:
    alpha_i = 1e-5
    filename = os.path.join(ESTIMATED_FOLDER, 'utigsp_gauss_ci_alpha=%.2e.txt,alpha_i=%.2e.txt' % (alpha, alpha_i))
    ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_i)
    if not os.path.exists(filename):
        est_dag, _ = unknown_target_igsp(
            setting_list,
            set(range(nnodes)),
            ci_tester,
            invariance_tester,
            nruns=10,
        )
        np.savetxt(filename, est_dag.to_amat())

# === RUN UNKNOWN TARGET IGSP WITH GAUSS CI AND TARGETS REMOVED
alpha = 3e-1
for alpha_i in tqdm([1e-100, 1e-50]):
    file = 'utigsp_gauss_ci_unknown_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_i)
    filename = os.path.join(ESTIMATED_FOLDER, file)
    ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_i)
    setting_list_removed = [{'known_interventions': []} for setting in setting_list]
    if not os.path.exists(filename):
        est_dag, learned_interventions = unknown_target_igsp(
            setting_list_removed,
            set(range(nnodes)),
            ci_tester,
            invariance_tester,
            nruns=10,
        )
        print(learned_interventions)
        json.dump(list(map(list, learned_interventions)), open(os.path.join(ESTIMATED_FOLDER, 'learned_interventions_' + file), 'w'))
        np.savetxt(filename, est_dag.to_amat())

# # === RUN UNKNOWN TARGET IGSP WITH HSIC
# for alpha in tqdm([4e-1]):
#     alpha_invariance = 1e-5
#     filename = os.path.join(ESTIMATED_FOLDER, 'utigsp_hsic_alpha=%.2e,alpha_i=%.2e.txt' % (alpha, alpha_invariance))
#     if not os.path.exists(filename):
#         est_dag, learned_interventions = unknown_target_igsp(
#             sample_dict[frozenset()],
#             setting_list,
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

# # === RUN IGSP WITH GAUSS CI
# for alpha in tqdm([6e-1, 7e-1]):
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

# # === RUN IGSP WITH HSIC
# for alpha in tqdm([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1]):
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

# # === RUN ICP
# for alpha in tqdm([1e-3, 1e-2, 5e-2, 1e-1, 3e-1]):
#     filename = os.path.join(ESTIMATED_FOLDER, 'icp_alpha=%.2e' % alpha)
#     if not os.path.exists(filename):
#         amat = run_icp(
#             sample_folder,
#             alpha,
#             nonlinear=True
#         )
#         np.savetxt(filename, amat)

# === RUN GIES
est_dags_gies = []
for lambda_ in [600, 700, 800, 900]:
    filename = os.path.join(ESTIMATED_FOLDER, 'gies_lambda=%.2e.txt' % lambda_)
    if not os.path.exists(filename):
        amat = run_gies(sample_folder, lambda_)
        np.savetxt(filename, amat)
#
shutil.rmtree(sample_folder)
