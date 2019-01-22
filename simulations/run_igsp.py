import argparse
import os
import numpy as np
import causaldag as cd
from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
import multiprocessing
from tqdm import tqdm

import sys
sys.path.append('..')
from config import PROJECT_FOLDER
from simulations.create_dags_and_samples import save_dags_and_samples, get_dag_samples, get_sample_folder, get_dag_folder


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

    parser.add_argument('--nruns', type=int, default=10)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--alpha_invariant', type=float)
    parser.add_argument('--pooled', type=str, default='auto')

    # === PARSE ARGUMENTS
    args = parser.parse_args()

    ndags = args.ndags
    nnodes = args.nnodes
    nneighbors = args.nneighbors

    nsamples = args.nsamples
    nsettings = args.nsettings
    num_known = args.num_known
    num_unknown = args.num_unknown
    intervention = args.intervention

    nruns = args.nruns
    depth = args.depth
    alpha = args.alpha
    alpha_invariant = args.alpha_invariant
    pooled = args.pooled

    # === CREATE DAGS AND SAMPLES: THIS MUST BE DONE THE SAME WAY EVERY TIME FOR THIS TO WORK
    save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention)
    sample_folders = [
        get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)
        for dag_num in range(ndags)
    ]


    def _run_igsp(dag_num):
        # === GENERATE FILENAME
        sample_folder = sample_folders[dag_num]
        alg_folder = os.path.join(sample_folder, 'estimates', 'igsp')
        os.makedirs(alg_folder, exist_ok=True)
        filename = os.path.join(alg_folder, 'nruns=%d,depth=%d,alpha=%.2e,alpha_invariant=%.2e' % (nruns, depth, alpha, alpha_invariant))

        # === RUN ALGORITHM
        if not os.path.exists(filename):
            obs_samples, setting_list, sample_dict = get_dag_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)

            if pooled == 'false':
                suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=nsamples)
            elif pooled == 'true':
                all_samples = np.concatenate((obs_samples, *[setting['samples'] for setting in setting_list]), axis=0)
                suffstat = dict(C=np.corrcoef(all_samples, rowvar=False), n=nsamples)
            elif nsamples <= 300:
                suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=nsamples)
            else:
                all_samples = np.concatenate((obs_samples, *[setting['samples'] for setting in setting_list]), axis=0)
                suffstat = dict(C=np.corrcoef(all_samples, rowvar=False), n=nsamples)

            est_dag = igsp(
                sample_dict,
                suffstat,
                nnodes,
                gauss_ci_test,
                hsic_invariance_test,
                alpha=alpha,
                alpha_invariance=alpha_invariant,
                depth=depth,
                nruns=nruns
            )

            np.savetxt(filename, est_dag.to_amat())
            return est_dag
        else:
            return cd.DAG.from_amat(np.loadtxt(filename))


    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        dag_nums = list(range(ndags))
        est_dags = list(tqdm(pool.imap(_run_igsp, dag_nums), total=ndags))

    # === CREATE FOLDER FOR RESULTS
    dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)
    sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention)
    alg_str = 'nruns=%d,depth=%d,alpha=%.2e,alpha_invariant=%.2e,pool=%s' % (nruns, depth, alpha, alpha_invariant, pooled)
    result_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, sample_str, 'igsp', alg_str)
    os.makedirs(result_folder, exist_ok=True)

    # === LOAD TRUE DAGS
    dag_filenames = [os.path.join(get_dag_folder(ndags, nnodes, nneighbors, dag_num), 'amat.txt') for dag_num in range(ndags)]
    true_dags = [cd.DAG.from_amat(np.loadtxt(dag_filename)) for dag_filename in dag_filenames]

    # === SAVE SHDS
    shds = [true_dag.shd(est_dag) for true_dag, est_dag in zip(true_dags, est_dags)]
    np.savetxt(os.path.join(result_folder, 'shds.txt'), shds)

    # === GET LISTS OF KNOWN AND ALL INTERVENTIONS
    def get_interventions(filename):
        known_iv_str, unknown_iv_str = filename.split(';')
        unknown_iv_str = unknown_iv_str.split('=')[1][:-4]
        known_ivs = set(map(int, known_iv_str.split('=')[1].split(',')))
        unknown_ivs = set(map(int, unknown_iv_str.split(','))) if unknown_iv_str != '' else set()
        return known_ivs, unknown_ivs

    intervention_filenames_list = [sorted(os.listdir(os.path.join(sample_folder, 'interventional'))) for sample_folder in sample_folders]
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
    np.savetxt(os.path.join(result_folder, 'same_icpdag.txt'), same_icpdag)

    shds_pdag = [est_pdag.shd(true_pdag) for est_pdag, true_pdag in zip(est_pdags, true_pdags)]
    np.savetxt(os.path.join(result_folder, 'shds_pdag.txt'), shds_pdag)

    is_imec = [
        true_dag.markov_equivalent(est_dag, interventions=true_interventions)
        for true_dag, est_dag, true_interventions in zip(true_dags, est_dags, true_interventions_list)
    ]
    np.savetxt(os.path.join(result_folder, 'imec.txt'), is_imec)
