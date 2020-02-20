import causaldag as cd
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test
from causaldag.utils.invariance_tests import MemoizedInvarianceTester, gauss_invariance_suffstat, gauss_invariance_test
from causaldag.structure_learning import jci_gsp
import random
import numpy as np

seed = random.randint(1, 1098789258)
print(seed)
seed = 387721040
random.seed(seed)
np.random.seed(seed)


def combined_gauss_ci_test(suffstat, i, j, cond_set=None, alpha=.01, alpha_inv=.01):
    cond_set = {c for c in cond_set if not isinstance(c, str)}
    if isinstance(i, int) and isinstance(j, int):
        return gauss_ci_test(suffstat['ci'], i, j, cond_set=cond_set, alpha=alpha)
    else:
        if isinstance(i, str):
            context = int(i[1:])
            node = j
        else:
            context = int(j[1:])
            node = i
        return gauss_invariance_test(suffstat['invariance'], context, node, cond_set=cond_set, alpha=alpha_inv)


nnodes = 5
nodes = set(range(nnodes))
nneighbors = 1.5
nsettings = 5
num_unknown_targets = 0
INTERVENTION = cd.GaussIntervention(1, .01)
d = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1))
g = cd.rand.rand_weights(d)
known_iv_list = random.sample(list(nodes), nsettings)
unknown_ivs_list = [random.sample(list(nodes - {known_iv}), num_unknown_targets) for known_iv in known_iv_list]
all_ivs_list = [{known_iv, *unknown_ivs} for known_iv, unknown_ivs in zip(known_iv_list, unknown_ivs_list)]

nsamples = 5000
obs_samples = g.sample(nsamples)
iv_samples_list = [g.sample_interventional({iv: INTERVENTION for iv in all_ivs}, nsamples) for all_ivs in all_ivs_list]

ci_suffstat = gauss_ci_suffstat(obs_samples)
inv_suffstat = gauss_invariance_suffstat(obs_samples, context_samples_list=iv_samples_list)
ci_tester = MemoizedCI_Tester(gauss_ci_test, ci_suffstat)
invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, inv_suffstat)

combined_ci_tester = MemoizedCI_Tester(
    combined_gauss_ci_test,
    dict(ci=ci_suffstat, invariance=inv_suffstat),
    alpha_inv=1e-5
)

est_dag, est_targets_list = jci_gsp(
    [dict(known_interventions={known_iv}) for known_iv in known_iv_list],
    nodes,
    combined_ci_tester,
    verbose=True
)

est_cpdag = est_dag.cpdag()
true_cpdag = d.cpdag()
est_icpdag = est_dag.interventional_cpdag(est_targets_list, est_cpdag)
true_icpdag = d.interventional_cpdag(all_ivs_list, true_cpdag)
print(est_cpdag.shd(true_cpdag), true_cpdag.shd(est_cpdag))
print(est_icpdag.shd(true_icpdag), true_icpdag.shd(est_icpdag))
print(f"False positives: {[est_targets - targets for est_targets, targets in zip(est_targets_list, all_ivs_list)]}")
print(f"False negatives: {[targets - est_targets for est_targets, targets in zip(est_targets_list, all_ivs_list)]}")
print(f"True DAG: {d}")
print(f"Estimated DAG: {est_dag}")
