from causaldag.inference.structural import igsp, unknown_target_igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
from real_data_analysis.sachs.sachs_meta import SACHS_DATA_FOLDER, true_dag
import os
import pandas as pd
import numpy as np
import random

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
setting_list = [
    {'known_interventions': iv_nodes, 'samples': samples}
    for iv_nodes, samples in sample_dict.items()
    if iv_nodes != frozenset()
]

nnodes = obs_samples.shape[1]
np.random.seed(1729)
random.seed(1729)
# starting_permutations = [random.sample(list(range(nnodes)), nnodes) for i in range(10)]
starting_permutations = [true_dag.topological_sort()]
# === RUN IGSP
igsp_dag = igsp(
    sample_dict,
    suffstat,
    nnodes,
    gauss_ci_test,
    hsic_invariance_test,
    alpha=1e-5,
    alpha_invariance=1e-5,
    depth=6,
    verbose=True,
    starting_permutations=starting_permutations
)

np.random.seed(1729)
random.seed(1729)
# === RUN UTIGSP
utigsp_dag = unknown_target_igsp(
    obs_samples,
    setting_list,
    suffstat,
    nnodes,
    gauss_ci_test,
    hsic_invariance_test,
    alpha=1e-5,
    alpha_invariance=1e-5,
    depth=6,
    verbose=True,
    starting_permutations=starting_permutations
)

print(true_dag.shd(utigsp_dag))
print(true_dag.shd(igsp_dag))
print(igsp_dag.shd(utigsp_dag))

