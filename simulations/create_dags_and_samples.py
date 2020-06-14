import numpy as np
import random
import causaldag as cd
import itertools as itr
import os
from config import PROJECT_FOLDER

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'simulations', 'data')


INTERVENTIONS = {
    'perfect1': cd.GaussIntervention(1, .01),
    'perfect2': cd.GaussIntervention(1, .1),
    'inhibitory1': cd.ScalingIntervention(.1, .2),
    'soft1': cd.ScalingIntervention(.1, .2, mean=1),
    'zero': cd.GaussIntervention(0, 1),
    'shift': cd.ShiftIntervention(2)
}


def get_dag_folder(ndags, nnodes, nneighbors, dag_num, nonlinear=False):
    nonlinear_str = '_nonlinear' if nonlinear else ''
    base_folder = os.path.join(DATA_FOLDER, f'nnodes={nnodes}_nneighbors={nneighbors}_ndags={ndags}{nonlinear_str}')
    return os.path.join(base_folder, f'dag{dag_num}')


def get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num, nonlinear=False):
    dag_folder = get_dag_folder(ndags, nnodes, nneighbors, dag_num, nonlinear=nonlinear)
    sample_str = f'nsamples={nsamples},num_known={num_known},num_unknown={num_unknown},nsettings={nsettings},intervention={intervention_str}'
    sample_folder = os.path.join(dag_folder, 'samples', sample_str)
    return sample_folder


def save_dags_and_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, nonlinear=False):
    # === IF THIS HAS ALREADY BEEN DONE, SKIP
    nonlinear_str = '_nonlinear' if nonlinear else ''
    base_folder = os.path.join(DATA_FOLDER, f'nnodes={nnodes}_nneighbors={nneighbors}_ndags={ndags}{nonlinear_str}')
    sample_str = f'nsamples={nsamples},num_known={num_known},num_unknown={num_unknown},nsettings={nsettings},intervention={intervention_str}'
    # dag0_folder = get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, 0, nonlinear=nonlinear)
    if os.path.exists(os.path.join(base_folder, 'dag0', sample_str)):
        return

    random.seed(1729)
    np.random.seed(1729)
    nodes = set(range(nnodes))
    intervention = INTERVENTIONS[intervention_str]

    # === GENERATE DAGS
    dags = cd.rand.directed_erdos(nnodes, exp_nbrs=nneighbors, size=ndags)
    if nonlinear:
        gdags = [cd.rand.rand_nn_functions(dag) for dag in dags]
    else:
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
            iv_samples = gdag.sample_interventional(interventions, nsamples)
            # if isinstance(intervention, cd.PerfectInterventionalDistribution):
            #     iv_samples = gdag.sample_interventional_perfect(interventions, nsamples)
            # elif isinstance(intervention, cd.SoftInterventionalDistribution):
            #     iv_samples = gdag.sample_interventional_soft(interventions, nsamples)
            # else:
            #     raise ValueError("intervention is not an InterventionalDistribution")
            settings_list.append({'known_interventions': known_ivs, 'samples': iv_samples, 'unknown_interventions': unknown_ivs})
            sample_dict[frozenset(known_ivs)] = iv_samples

        samples_by_dag.append((obs_samples, settings_list))

    # === SAVE DAGS AND SAMPLES
    for dag_num, dag in enumerate(dags):
        # === SAVE DAG
        obs_samples, settings_list = samples_by_dag[dag_num]
        dag_folder = os.path.join(base_folder, f'dag{dag_num}')
        os.makedirs(dag_folder, exist_ok=True)
        np.save(os.path.join(dag_folder, 'amat.npy'), dag.to_amat()[0])

        # === SAVE SAMPLES
        sample_folder = os.path.join(dag_folder, 'samples', sample_str)
        iv_sample_folder = os.path.join(sample_folder, 'interventional')
        os.makedirs(iv_sample_folder, exist_ok=True)
        np.save(os.path.join(sample_folder, 'observational.npy'), obs_samples)
        for setting_num, setting in enumerate(settings_list):
            known_iv_str = ','.join(map(str, setting['known_interventions']))
            unknown_iv_str = ','.join(map(str, setting['unknown_interventions']))
            filename = os.path.join(iv_sample_folder, f'{setting_num}_known_ivs={known_iv_str};unknown_ivs={unknown_iv_str}.npy')
            np.save(filename, setting['samples'])


def get_dag_samples(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num, nonlinear=False):
    sample_folder = get_sample_folder(ndags, nnodes, nneighbors, nsamples, nsettings, num_known, num_unknown, intervention_str, dag_num, nonlinear=nonlinear)
    iv_sample_folder = os.path.join(sample_folder, 'interventional')

    obs_samples = np.load(os.path.join(sample_folder, 'observational.npy'))
    sample_dict = {frozenset(): obs_samples}
    setting_list = []
    for file in sorted(os.listdir(iv_sample_folder)):
        known_ivs = frozenset(map(int, file.split(';')[0].split('=')[1].split(',')))
        iv_samples = np.load(os.path.join(iv_sample_folder, file))
        sample_dict[known_ivs] = iv_samples
        setting_list.append({'known_interventions': known_ivs, 'samples': iv_samples})

    return obs_samples, setting_list, sample_dict




