import causaldag as cd
import random
import itertools as itr
import numpy as np
np.random.seed(1729)
random.seed(1729)

num_known = 1
num_unknown = 1
nsettings = 2
ndags = 1000
nnodes = 10
nneighbors = 5
nodes = set(range(nnodes))
dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
known_ivs_list = [random.sample(list(itr.combinations(nodes, num_known)), nsettings) for _ in range(ndags)]
known_ivs_list = [list(map(set, known_ivs)) for known_ivs in known_ivs_list]
unknown_ivs_list = [
    [set(random.sample(nodes - iv_nodes, num_unknown)) for iv_nodes in known_ivs]
    for known_ivs in known_ivs_list
]
full_ivs_list = [
    [known_iv_nodes | unknown_iv_nodes for known_iv_nodes, unknown_iv_nodes in zip(known_ivs, unknown_ivs)]
    for known_ivs, unknown_ivs in zip(known_ivs_list, unknown_ivs_list)
]
icpdags_known = [d.interventional_cpdag(known_ivs, cpdag=d.cpdag()) for d, known_ivs in zip(dags, known_ivs_list)]
icpdags_all = [d.interventional_cpdag(full_ivs, cpdag=d.cpdag()) for d, full_ivs in zip(dags, full_ivs_list)]
same = [icpdag_known == icpdag_all for icpdag_known, icpdag_all in zip(icpdags_known, icpdags_all)]
print(sum(same))