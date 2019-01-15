import causaldag as cd
from causaldag import GaussIntervention
from causaldag.utils.ci_tests import hsic_test_vector
import numpy as np
import os

g = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
s = g.sample(200)
np.savetxt(os.path.expanduser('~/Desktop/s.txt'), s)
r = hsic_test_vector(s[:, 1], s[:, 2], sig=1/np.sqrt(2))
print(r)
