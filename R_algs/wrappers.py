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


def run_igsp(sample_folder, alpha):
    r_file = os.path.join(PROJECT_FOLDER, 'R_algs', 'run_igsp.R')
    r_command = 'Rscript "%s" %s %s' % (r_file, alpha, sample_folder)
    os.makedirs(os.path.join(sample_folder, 'estimates', 'igsp-r'), exist_ok=True)
    os.system(r_command)
    amat = np.loadtxt(os.path.join(sample_folder, 'estimates', 'igsp-r', 'alpha=%.2e.txt' % alpha))
    return amat


def run_icp(sample_folder, alpha):
    r_file = os.path.join(PROJECT_FOLDER, 'R_algs', 'run_icp.R')
    r_command = 'Rscript "%s" %s %s' % (r_file, alpha, sample_folder)
    os.makedirs(os.path.join(sample_folder, 'estimates', 'icp'), exist_ok=True)
    os.system(r_command)
    amat = np.loadtxt(os.path.join(sample_folder, 'estimates', 'icp', 'alpha=%.2e.txt' % alpha))
    return amat

