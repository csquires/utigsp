import causaldag as cd
import time
import numpy as np

g = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
cov = g.covariance
nsamples = 500
trials = 10
iv = {frozenset({1}): cd.GaussIntervention(0, 1)}

# === TIME INVARIANCE TEST
start = time.time()
for _ in range(trials):
    samples = g.sample(nsamples)
    samples_iv = g.sample_interventional(iv, nsamples)
    cd.utils.ci_tests.kci_invariance_test(samples, samples_iv, 0, 1)
print(time.time() - start)

start = time.time()
for _ in range(trials):
    samples = g.sample(nsamples)
    samples_iv = g.sample_interventional(iv, nsamples)
    cd.utils.ci_tests.kci_invariance_test(samples, samples_iv, 0, 1, regress=False)
print(time.time() - start)

# === TIME CI TEST
start = time.time()
for _ in range(trials):
    samples = g.sample(nsamples*2)
    cd.utils.ci_tests.kci_test(samples, 1, 2, 0)
print(time.time() - start)
