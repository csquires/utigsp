import argparse
import os
import numpy as np
import causaldag as cd
import multiprocessing
from tqdm import tqdm

import sys
sys.path.append('..')
from R_algs.wrappers import run_gies
from config import PROJECT_FOLDER
from simulations.create_dags_and_samples import save_dags_and_samples, get_dag_samples, get_sample_folder, get_dag_folder

overwrite = False

if __name__ == '__main__':
    # === DEFINE PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--ndags', type=int)
    parser.add_argument('--nneighbors', type=float)

    parser.add_argument('--nsamples', type=int)
    parser.add_argument('--nsettings', type=int)
    parser.add_argument('--num_known', type=int)
    parser.add_argument('--num_unknown', type=int)
    parser.add_argument('--intervention', type=str)

    parser.add_argument('--lam', type=float, default=1)

    # === PARSE ARGUMENTS
    args = parser.parse_args()
    print(args)

    ndags = args.ndags
    nnodes = args.nnodes
    nneighbors = args.nneighbors

    nsamples = args.nsamples
    nsettings = args.nsettings
    num_known = args.num_known
    num_unknown = args.num_unknown
    intervention = args.intervention

    lam = args.lam

    # === CREATE DAGS AND SAMPLES: THIS MUST BE DONE THE SAME WAY EVERY TIME FOR THIS TO WORK
    save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention)
    sample_folders = [
        get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)
        for dag_num in range(ndags)
    ]


    def _run_gies(dag_num):
        # === GENERATE FILENAME
        sample_folder = sample_folders[dag_num]
        alg_folder = os.path.join(sample_folder, 'estimates', 'gies')
        os.makedirs(alg_folder, exist_ok=True)
        filename = os.path.join(alg_folder, 'lambda_=%.2e' % lam)

        # === RUN ALGORITHM
        if not os.path.exists(filename) or overwrite:
            est_amat = run_gies(
                sample_folder,
                lambda_=lam
            )

            np.savetxt(filename, est_amat)
            return cd.DAG.from_amat(est_amat)
        else:
            return cd.DAG.from_amat(np.loadtxt(filename))


    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        dag_nums = list(range(ndags))
        est_dags = list(tqdm(pool.imap(_run_gies, dag_nums), total=ndags))

    # === CREATE FOLDER FOR RESULTS
    dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)
    sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention)
    alg_str = 'lambda_=%.2e' % lam
    result_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, sample_str, 'gies', alg_str)
    os.makedirs(result_folder, exist_ok=True)

    # === LOAD TRUE DAGS
    dag_filenames = [os.path.join(get_dag_folder(ndags, nnodes, nneighbors, dag_num), 'amat.txt') for dag_num in range(ndags)]
    true_dags = [cd.DAG.from_amat(np.loadtxt(dag_filename)) for dag_filename in dag_filenames]

    # === SAVE SHDS
    shds = [true_dag.shd(est_dag) for true_dag, est_dag in zip(true_dags, est_dags)]
    np.savetxt(os.path.join(result_folder, 'shds.txt'), shds)

    # === SAVE IS-IMEC
    def get_interventions(filename):
        known_iv_str, unknown_iv_str = filename.split(';')
        known_ivs = set(map(int, known_iv_str.split('=')[1].split(',')))
        unknown_ivs = set(map(int, known_iv_str.split('=')[1].split(',')))
        return known_ivs | unknown_ivs


    intervention_filenames_list = [os.listdir(os.path.join(sample_folder, 'interventional')) for sample_folder in sample_folders]
    interventions_list = [
        [get_interventions(filename) for filename in intervention_filenames]
        for intervention_filenames in intervention_filenames_list
    ]
    is_imec = [
        true_dag.markov_equivalent(est_dag, interventions=interventions)
        for true_dag, est_dag, interventions in zip(true_dags, est_dags, interventions_list)
    ]
    np.savetxt(os.path.join(result_folder, 'imec.txt'), is_imec)


