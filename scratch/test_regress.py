from pygam import GAM
import causaldag as cd
import numpy as np
import os
import random
np.random.seed(1729)
random.seed(1729)

d = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
s = d.sample(100)
np.savetxt(os.path.expanduser('~/Desktop/s1.txt'), s)

gam = GAM()
gam.fit(s[:, 0], s[:, 1])
res1 = gam.deviance_residuals(s[:, 0], s[:, 1])
print(gam.summary())
gam.fit(s[:, 0], s[:, 1])
res2 = gam.deviance_residuals(s[:, 0], s[:, 2])
print(gam.summary())
print(res1)
print(res2)
