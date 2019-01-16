import causaldag as cd
from causaldag import GaussIntervention
from causaldag.inference.structural import igsp, unknown_target_igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
import numpy as np
import random
import os
from config import PROJECT_FOLDER
from R_algs.wrappers import run_gies
np.random.seed(1729)
random.seed(1729)

ntrials = 10
nnodes = 5
d = cd.DAG(arcs={(i, i+1) for i in range(nnodes-1)})
g = cd.GaussDAG(nodes=list(range(nnodes)), arcs=d.arcs)
cpdag = d.cpdag()
print(d.interventional_cpdag({nnodes-1}, cpdag=cpdag).arcs)
print(d.interventional_cpdag({0, nnodes-1}, cpdag=cpdag).arcs)

shds_igsp = []
shds_utigsp = []
shds_gies = []
dags_igsp = []
dags_utigsp = []
dags_gies = []
for i in range(ntrials):
    nsamples = 500
    intervention = GaussIntervention(1, .01)
    samples = g.sample(nsamples)
    iv_samples = g.sample_interventional_perfect({0: intervention, nnodes-1: intervention}, nsamples)
    sample_dict = {frozenset(): samples, frozenset({nnodes-1}): iv_samples}
    suffstat = dict(C=np.corrcoef(samples, rowvar=False), n=nsamples)

    # === SAVE SAMPLES
    sample_folder = os.path.join(PROJECT_FOLDER, 'tmp')
    iv_sample_folder = os.path.join(sample_folder, 'interventional')
    os.makedirs(iv_sample_folder, exist_ok=True)
    np.savetxt(os.path.join(sample_folder, 'observational.txt'), samples)
    iv_str = 'known_ivs=%s;unknown_ivs=%s.txt' % (nnodes-1, 0)
    np.savetxt(os.path.join(iv_sample_folder, iv_str), iv_samples)

    # === RUN ALGORITHMS
    dag_igsp = igsp(sample_dict, suffstat, nnodes, gauss_ci_test, hsic_invariance_test, nruns=10, alpha_invariance=1e-5)
    dag_utigsp = unknown_target_igsp(sample_dict, suffstat, nnodes, gauss_ci_test, hsic_invariance_test, nruns=10, alpha_invariance=1e-2)
    amat_gies = run_gies(sample_folder, lambda_=50)
    dag_gies = cd.DAG.from_amat(amat_gies)

    dags_igsp.append(dag_igsp)
    dags_utigsp.append(dag_utigsp)
    dags_gies.append(dag_gies)
    shds_igsp.append(dag_igsp.shd(d))
    shds_utigsp.append(dag_utigsp.shd(d))
    shds_gies.append(dag_gies.shd(d))

print(np.mean(shds_igsp))
print(np.mean(shds_utigsp))
print(np.mean(shds_gies))
