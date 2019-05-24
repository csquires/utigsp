import argparse
import os
import numpy as np
import causaldag as cd
from causaldag.inference.structural import jci_gsp
from causaldag.utils.ci_tests import gauss_ci_test, gauss_ci_suffstat, MemoizedCI_Tester
from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
import multiprocessing
from tqdm import tqdm

import sys
sys.path.append('..')
from config import PROJECT_FOLDER
from simulations.create_dags_and_samples import save_dags_and_samples, get_dag_samples, get_sample_folder, get_dag_folder
import json
from pprint import pprint

overwrite = True


def combined_gauss_ci_test(suffstat, i, j, cond_set=None, alpha=.01, alpha_inv=.01):
    cond_set = {c for c in cond_set if not isinstance(c, str)}
    if isinstance(i, int) and isinstance(j, int):
        return gauss_ci_test(suffstat['ci'], i, j, cond_set=cond_set, alpha=alpha)
    else:
        if isinstance(i, str):
            context = int(i[1:])
            node = j
        else:
            context = int(j[1:])
            node = i
        return gauss_invariance_test(suffstat['invariance'], context, node, cond_set=cond_set, alpha=alpha_inv)


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
    nodes = set(range(nnodes))
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


    def _run_jcigsp(dag_num):
        # === GENERATE FILENAME
        sample_folder = sample_folders[dag_num]
        alg_folder = os.path.join(sample_folder, 'estimates', 'jcigsp')
        os.makedirs(alg_folder, exist_ok=True)
        filename = os.path.join(alg_folder, 'nruns=%d,depth=%d,alpha=%.2e,alpha_invariant=%.2e.npy' % (nruns, depth, alpha, alpha_invariant))

        # === RUN ALGORITHM
        if not os.path.exists(filename) or overwrite:
            obs_samples, setting_list, _ = get_dag_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention, dag_num)

            suffstat = gauss_ci_suffstat(obs_samples)
            suffstat_inv = gauss_invariance_suffstat(obs_samples, [setting['samples'] for setting in setting_list])
            combined_ci_tester = MemoizedCI_Tester(
                combined_gauss_ci_test,
                dict(ci=suffstat, invariance=suffstat_inv),
                alpha=alpha,
                alpha_inv=alpha_invariant
            )

            est_dag, learned_intervention_targets = jci_gsp(
                [{'known_interventions': setting['known_interventions']} for setting in setting_list],
                nodes,
                combined_ci_tester,
                depth=depth,
                nruns=nruns
            )

            np.save(filename, est_dag.to_amat()[0])
            json.dump(list(map(list, learned_intervention_targets)), open(filename + '_learned_intervention_targets.json', 'w'))
            return est_dag, learned_intervention_targets
        else:
            learned_intervention_targets = json.load(open(filename + '_learned_intervention_targets.json'))
            return cd.DAG.from_amat(np.load(filename)), learned_intervention_targets


    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        dag_nums = list(range(ndags))
        est_dags_and_iv_targets = list(tqdm(pool.imap(_run_jcigsp, dag_nums), total=ndags))
    est_dags, est_interventions_list = zip(*est_dags_and_iv_targets)

    # === CREATE FOLDER FOR RESULTS
    dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)
    sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention)
    alg_str = 'nruns=%d,depth=%d,alpha=%.2e,alpha_invariant=%.2e,pool=%s' % (nruns, depth, alpha, alpha_invariant, pooled)
    result_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, sample_str, 'jcigsp', alg_str)
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


    intervention_filenames_list = [sorted(os.listdir(os.path.join(sample_folder, 'interventional'))) for sample_folder in sample_folders]
    known_interventions_list = [
        [get_interventions(filename)[0] for filename in intervention_filenames]
        for intervention_filenames in intervention_filenames_list
    ]
    true_interventions_list = [
        [get_interventions(filename)[0] | get_interventions(filename)[1] for filename in intervention_filenames]
        for intervention_filenames in intervention_filenames_list
    ]
    print(true_interventions_list[0])
    print(est_interventions_list[0])

    # === FIND ESTIMATED PDAGS
    est_pdags = [
        dag.interventional_cpdag(
            [set(est_iv_nodes) | set(known_iv_nodes) for est_iv_nodes, known_iv_nodes in zip(est_ivs, known_ivs)],
            cpdag=dag.cpdag()
        )
        for dag, est_ivs, known_ivs in zip(est_dags, est_interventions_list, known_interventions_list)
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

    # === DIFFERENCE BETWEEN LEARNED AND TRUE INTERVENTIONS
    difference_interventions = [
        [
            len(set(true_iv_nodes) - set(est_iv_nodes)) + len(set(est_iv_nodes) - set(true_iv_nodes))
            for true_iv_nodes, est_iv_nodes in zip(true_interventions, est_interventions)
         ]
        for true_interventions, est_interventions in zip(true_interventions_list, est_interventions_list)
    ]
    missing_interventions = [
        [
            len(set(true_iv_nodes) - set(est_iv_nodes))
            for true_iv_nodes, est_iv_nodes in zip(true_interventions, est_interventions)
        ]
        for true_interventions, est_interventions in zip(true_interventions_list, est_interventions_list)
    ]
    added_interventions = [
        [
            len(set(est_iv_nodes) - set(true_iv_nodes))
            for true_iv_nodes, est_iv_nodes in zip(true_interventions, est_interventions)
        ]
        for true_interventions, est_interventions in zip(true_interventions_list, est_interventions_list)
    ]
    np.save(os.path.join(result_folder, 'diff_interventions.npy'), difference_interventions)
    np.save(os.path.join(result_folder, 'missing_interventions.npy'), missing_interventions)
    np.save(os.path.join(result_folder, 'added_interventions.npy'), added_interventions)



