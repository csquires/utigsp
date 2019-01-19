import argparse
import os
import sys
sys.path.append('../..')
from R_algs.wrappers import run_gies
from config import PROJECT_FOLDER
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

# === LOAD DATA
sample_dict, _ = get_sample_dict()
sample_dict_exclude = {k: v for k, v in sample_dict.items() if k != frozenset({excluded})}
obs_samples = sample_dict[frozenset()]

# === RUN ALGORITHM
if not os.path.exists(filename):
    sample_folder = os.path.join(PROJECT_FOLDER, 'tmp_dixit_iv=%s' % excluded)
    iv_sample_folder = os.path.join(sample_folder, 'interventional')
    os.makedirs(iv_sample_folder, exist_ok=True)
    np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
    for iv_nodes, samples in sample_dict_exclude.items():
        if iv_nodes != frozenset():
            iv_str = 'known_ivs=%s;unknown_ivs=.txt' % ','.join(map(str, iv_nodes))
            np.savetxt(os.path.join(iv_sample_folder, iv_str), samples)
    est_amat = run_gies(
        sample_folder,
        lam
    )
    np.savetxt(filename, est_amat)

