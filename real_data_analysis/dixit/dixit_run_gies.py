import argparse
import os
from R_algs.wrappers import run_gies
from real_data_analysis.dixit.dixit_meta import get_sample_dict, ESTIMATED_FOLDER, nnodes
import numpy as np

# === PARSE
parser = argparse.ArgumentParser
parser.add_argument('--lam', type=float)
parser.add_argument('--excluded', type=int)
args = parser.parse_args()
lam = args.lam
excluded = args.excluded

# === CREATE FILENAME
iv_estimated_folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%s' % excluded)
filename = os.path.join(iv_estimated_folder, 'gies_lambda=%.2e.txt' % lam)

# === RUN ALGORITHM
if not os.path.exists(filename):
    est_amat = run_gies(
        sample_folder,
        lam
    )
    np.savetxt(filename, est_amat)

