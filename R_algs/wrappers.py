import os
import sys
import numpy as np
from config import PROJECT_FOLDER


def run_gies(sample_folder, lambda_):
    r_file = os.path.join(PROJECT_FOLDER, 'R_algs', 'run_gies.R')
    r_command = 'Rscript "%s" %s %s' % (r_file, lambda_, sample_folder)
    os.makedirs(os.path.join(sample_folder, 'estimates', 'gies'), exist_ok=True)
    os.system(r_command)
    amat = np.loadtxt(os.path.join(sample_folder, 'estimates', 'gies', 'lambda=%.2e.txt' % lambda_))
    return amat


def run_icp(sample_folder):
    raise NotImplementedError
