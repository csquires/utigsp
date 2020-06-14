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
MULTITHREAD = False

if __name__ == '__main__':
    # === DEFINE PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--ndags', type=int)
    parser.add_argument('--nneighbors', type=float)
    parser.add_argument('--nonlinear', type=str)

    parser.add_argument('--nsamples', type=int)
    parser.add_argument('--nsettings', type=int)
    parser.add_argument('--num_known', type=int)
    parser.add_argument('--num_unknown', type=int)
    parser.add_argument('--intervention', type=str)

    parser.add_argument('--lam', type=float, default=1)

    # === PARSE ARGUMENTS
    args = parser.parse_args()
    print(args, 'GIES')

    ndags = args.ndags
    nnodes = args.nnodes
    nneighbors = args.nneighbors
    nonlinear = args.nonlinear == 'True'

    nsamples = args.nsamples
    nsettings = args.nsettings
    num_known = args.num_known
    num_unknown = args.num_unknown
    intervention = args.intervention

    lam = args.lam

    # === CREATE DAGS AND SAMPLES: THIS MUST BE DONE THE SAME WAY EVERY TIME FOR THIS TO WORK
    save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, nonlinear=nonlinear)
    sample_folders = [
        get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num, nonlinear=nonlinear)
        for dag_num in range(ndags)
    ]


    def _run_gies(dag_num):
        # === GENERATE FILENAME
        sample_folder = sample_folders[dag_num]
        alg_folder = os.path.join(sample_folder, 'estimates', 'gies')
        os.makedirs(alg_folder, exist_ok=True)
        filename = os.path.join(alg_folder, 'lambda_=%.2e.npy' % lam)

        # === RUN ALGORITHM
        if not os.path.exists(filename) or overwrite:
            est_amat = run_gies(
                sample_folder,
                lambda_=lam
            )

            np.save(filename, est_amat)
            return cd.DAG.from_amat(est_amat)
        else:
            return cd.DAG.from_amat(np.load(filename))

    if MULTITHREAD:
        with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
            dag_nums = list(range(ndags))
            est_dags = list(tqdm(pool.imap(_run_gies, dag_nums), total=ndags))
    else:
        est_dags = list(tqdm((_run_gies(dag_num) for dag_num in range(ndags)), total=ndags))

    # === CREATE FOLDER FOR RESULTS
    nonlinear_str = '_nonlinear' if nonlinear else ''
    dag_str = f'nnodes={nnodes}_nneighbors={nneighbors}_ndags={ndags}{nonlinear_str}'
    sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention)
    alg_str = 'lambda_=%.2e' % lam
    result_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, sample_str, 'gies', alg_str)
    os.makedirs(result_folder, exist_ok=True)

    # === LOAD TRUE DAGS
    dag_filenames = [os.path.join(get_dag_folder(ndags, nnodes, nneighbors, dag_num), 'amat.npy') for dag_num in range(ndags)]
    true_dags = [cd.DAG.from_amat(np.load(dag_filename)) for dag_filename in dag_filenames]

    # === SAVE SHDS
    shds = [true_dag.shd(est_dag) for true_dag, est_dag in zip(true_dags, est_dags)]
    np.save(os.path.join(result_folder, 'shds.npy'), shds)

    # === GET LISTS OF KNOWN AND ALL INTERVENTIONS
    def get_interventions(filename):
        known_iv_str, unknown_iv_str = filename.split(';')
        unknown_iv_str = unknown_iv_str.split('=')[1][:-4]
        known_ivs = set(map(int, known_iv_str.split('=')[1].split(',')))
        unknown_ivs = set(map(int, unknown_iv_str.split(','))) if unknown_iv_str != '' else set()
        return known_ivs, unknown_ivs


    intervention_filenames_list = [sorted(os.listdir(os.path.join(sample_folder, 'interventional'))) for sample_folder
                                   in sample_folders]
    known_interventions_list = [
        [get_interventions(filename)[0] for filename in intervention_filenames]
        for intervention_filenames in intervention_filenames_list
    ]
    true_interventions_list = [
        [get_interventions(filename)[0] | get_interventions(filename)[1] for filename in intervention_filenames]
        for intervention_filenames in intervention_filenames_list
    ]

    # === FIND ESTIMATED PDAGS
    est_pdags = [
        dag.interventional_cpdag(known_ivs, cpdag=dag.cpdag())
        for dag, known_ivs in zip(est_dags, known_interventions_list)
    ]

    # === FIND TRUE PDAGS
    true_pdags = [
        true_dag.interventional_cpdag(true_interventions, cpdag=true_dag.cpdag())
        for true_dag, true_interventions in zip(true_dags, true_interventions_list)
    ]

    # === COMPARE TRUE PDAGS TO ESTIMATED PDAGS
    same_icpdag = [est_pdag == true_pdag for est_pdag, true_pdag in zip(est_pdags, true_pdags)]
    np.save(os.path.join(result_folder, 'same_icpdag.npy'), same_icpdag)

    shds_pdag = [est_pdag.shd(true_pdag) for est_pdag, true_pdag in zip(est_pdags, true_pdags)]
    np.save(os.path.join(result_folder, 'shds_pdag.npy'), shds_pdag)

    is_imec = [
        true_dag.markov_equivalent(est_dag, interventions=true_interventions)
        for true_dag, est_dag, true_interventions in zip(true_dags, est_dags, true_interventions_list)
    ]
    np.save(os.path.join(result_folder, 'imec.npy'), is_imec)

