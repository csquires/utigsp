import numpy as np
import random
import causaldag as cd
import itertools as itr
import os
from config import PROJECT_FOLDER


INTERVENTIONS = {'perfect1': cd.GaussIntervention(1, .01), 'inhibitory1': cd.ScalingIntervention(.1, .2)}


def get_dag_folder(ndags, nnodes, nneighbors, dag_num):
    base_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'data', 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags))
    return os.path.join(base_folder, 'dag%d' % dag_num)


def get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num):
    dag_folder = get_dag_folder(ndags, nnodes, nneighbors, dag_num)
    sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention_str)
    sample_folder = os.path.join(dag_folder, 'samples', sample_str)
    return sample_folder


def save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str):
    # === IF THIS HAS ALREADY BEEN DONE, SKIP
    if os.path.exists(os.path.join(
        PROJECT_FOLDER, 'simulations', 'data',
        'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags), 'dag0',
        'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention_str)
    )):
        return

    random.seed(1729)
    np.random.seed(1729)
    nodes = set(range(nnodes))
    intervention = INTERVENTIONS[intervention_str]

    # === GENERATE DAGS
    dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes - 1), ndags)
    gdags = [cd.rand.rand_weights(dag) for dag in dags]

    # === GENERATE SAMPLES
    samples_by_dag = []
    for gdag in gdags:
        sample_dict = {}
        settings_list = []
        obs_samples = gdag.sample(nsamples)
        sample_dict[frozenset()] = obs_samples

        # === PICK INTERVENTIONS
        possible_known_ivs = list(itr.combinations(nodes, num_known))
        known_ivs_list = random.sample(possible_known_ivs, nsettings)
        unknown_ivs_list = [random.sample(nodes - set(known_ivs), num_unknown) for known_ivs in known_ivs_list]

        # === GENERATE INTERVENTIONAL DATA
        for known_ivs, unknown_ivs in zip(known_ivs_list, unknown_ivs_list):
            interventions = {iv_node: intervention for iv_node in set(known_ivs) | set(unknown_ivs)}
            if isinstance(intervention, cd.PerfectInterventionalDistribution):
                iv_samples = gdag.sample_interventional_perfect(interventions, nsamples)
            elif isinstance(intervention, cd.SoftInterventionalDistribution):
                iv_samples = gdag.sample_interventional_soft(interventions, nsamples)
            else:
                raise ValueError("intervention is not an InterventionalDistribution")
            settings_list.append({'known_interventions': known_ivs, 'samples': iv_samples, 'unknown_interventions': unknown_ivs})
            sample_dict[frozenset(known_ivs)] = iv_samples

        samples_by_dag.append((obs_samples, settings_list))

    # === SAVE DAGS AND SAMPLES
    base_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'data', 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags))
    for dag_num, dag in enumerate(dags):
        # === SAVE DAG
        obs_samples, settings_list = samples_by_dag[dag_num]
        dag_folder = os.path.join(base_folder, 'dag%d' % dag_num)
        os.makedirs(dag_folder, exist_ok=True)
        np.savetxt(os.path.join(dag_folder, 'amat.txt'), dag.to_amat())

        # === SAVE SAMPLES
        sample_str = 'nsamples=%s,num_known=%d,num_unknown=%d,nsettings=%d,intervention=%s' % (nsamples, num_known, num_unknown, nsettings, intervention_str)
        sample_folder = os.path.join(dag_folder, 'samples', sample_str)
        iv_sample_folder = os.path.join(sample_folder, 'interventional')
        os.makedirs(iv_sample_folder, exist_ok=True)
        np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
        for setting_num, setting in enumerate(settings_list):
            known_iv_str = ','.join(map(str, setting['known_interventions']))
            unknown_iv_str = ','.join(map(str, setting['unknown_interventions']))
            filename = os.path.join(iv_sample_folder, '%d_known_ivs=%s;unknown_ivs=%s.txt' % (setting_num, known_iv_str, unknown_iv_str))
            np.savetxt(filename, setting['samples'])


def get_dag_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num):
    sample_folder = get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num)
    iv_sample_folder = os.path.join(sample_folder, 'interventional')

    obs_samples = np.loadtxt(os.path.join(sample_folder, 'observational.txt'))
    sample_dict = {frozenset(): obs_samples}
    setting_list = []
    for file in sorted(os.listdir(iv_sample_folder)):
        known_ivs = frozenset(map(int, file.split(';')[0].split('=')[1].split(',')))
        iv_samples = np.loadtxt(os.path.join(iv_sample_folder, file))
        sample_dict[known_ivs] = iv_samples
        setting_list.append({'known_interventions': known_ivs, 'samples': iv_samples})

    return obs_samples, setting_list, sample_dict




