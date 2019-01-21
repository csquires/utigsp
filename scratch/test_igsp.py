from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
import causaldag as cd
import numpy as np
import random

np.random.seed(40)
random.seed(9879132)

nnodes = 10
nsamples = 100
dag = cd.rand.directed_erdos(nnodes, 1.5/(nnodes-1), 1)
gdag = cd.rand.rand_weights(dag)
obs_samples = gdag.sample(nsamples)
sample_dict = {}
sample_dict[frozenset()] = obs_samples
for i in range(10):
    sample_dict[frozenset({i})] = gdag.sample_interventional_perfect({i: cd.GaussIntervention(1, .1)}, nsamples)
suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=nsamples)

est_dag = igsp(
    sample_dict,
    suffstat,
    nnodes,
    gauss_ci_test,
    hsic_invariance_test,
    1e-5,
    1e-5,
    nruns=5,
    verbose=True
)