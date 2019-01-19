from R_algs.wrappers import run_icp
import causaldag as cd
import os
import numpy as np
from config import PROJECT_FOLDER

nsamples = 10
g = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (1, 2)})
obs_samples = g.sample(nsamples)
iv_node = 1
iv_samples = g.sample_interventional_perfect({iv_node: cd.GaussIntervention(10, .01)}, nsamples)

# === SAVE DATA
sample_folder = os.path.join(PROJECT_FOLDER, 'tmp_icp_test')
iv_sample_folder = os.path.join(sample_folder, 'interventional')
os.makedirs(iv_sample_folder, exist_ok=True)
np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
np.savetxt(os.path.join(iv_sample_folder, 'known_ivs=%s;unknown_ivs=.txt' % iv_node), iv_samples)

# === RUN ICP
run_icp(sample_folder, .01)

