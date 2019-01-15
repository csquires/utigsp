import causaldag as cd
import time
import numpy as np

g = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
cov = g.covariance
nsamples = 2500
trials = 10
iv = {1: cd.GaussIntervention(0, 1)}

# === TIME INVARIANCE TEST NO CONDITIONING SET
start = time.time()
for _ in range(trials):
    samples = g.sample(nsamples)
    cd.utils.ci_tests.hsic_test_vector(samples[:, 0], samples[:, 1])
print(time.time() - start)

# === TIME INVARIANCE TEST WITH CONDITIONING SET
# start = time.time()
# for _ in range(trials):
#     samples = g.sample(nsamples)
#     cd.utils.ci_tests.hsic_invariance_test(samples[:, 0], samples[:, 1], 0)
# print(time.time() - start)

