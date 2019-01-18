from R_algs.wrappers import run_igsp
import causaldag as cd
import numpy as np
import random
import os
from tqdm import tqdm
from config import PROJECT_FOLDER
np.random.seed(1729)
random.seed(1729)

nnodes = 20
nneighbors = 1.5
ndags = 100
nsamples = 100
intervention = cd.ScalingIntervention(.1, .2)
dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
gdags = [cd.rand.rand_weights(d) for d in dags]
samples_by_dag = [{
    frozenset(): g.sample(nsamples),
    **{frozenset({i}): g.sample_interventional_soft({i: intervention}) for i in range(nnodes)}
} for g in gdags]

est_amats = []
# === SAVE SAMPLES
for sample_dict in tqdm(samples_by_dag):
    sample_folder = os.path.join(PROJECT_FOLDER, 'tmp_test_igsp')
    iv_sample_folder = os.path.join(sample_folder, 'interventional')
    os.makedirs(iv_sample_folder, exist_ok=True)
    np.savetxt(os.path.join(sample_folder, 'observational.txt'), sample_dict[frozenset()])
    for i in range(nnodes):
        np.savetxt(os.path.join(iv_sample_folder, 'known_ivs=%s;unknown_ivs=.txt' % i), sample_dict[frozenset({i})])
    est_amat = run_igsp(sample_folder, 1e-5)
    est_amats.append(est_amat)

est_dags = [cd.DAG.from_amat(a) for a in est_amats]
shds = [true_dag.shd(est_dag) for true_dag, est_dag in zip(dags, est_dags)]

