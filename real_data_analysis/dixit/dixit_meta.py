import os
from config import PROJECT_FOLDER
import numpy as np
import pandas as pd
import random

DIXIT_FOLDER = os.path.join(PROJECT_FOLDER, 'real_data_analysis', 'dixit')
DIXIT_DATA_FOLDER = os.path.join(DIXIT_FOLDER, 'data')
ESTIMATED_FOLDER = os.path.join(DIXIT_FOLDER, 'estimated')
FIGURES_FOLDER = os.path.join(DIXIT_FOLDER, 'figures')
nnodes = 24
for i in range(24):
    os.makedirs(os.path.join(ESTIMATED_FOLDER, 'exclude_%d' % i), exist_ok=True)


def get_sample_dict():
    sample_dict = {}
    ivs = []
    for file in os.listdir(DIXIT_DATA_FOLDER):
        samples = pd.read_csv(os.path.join(DIXIT_DATA_FOLDER, file), sep=',')
        iv_str = file.split('=')[1][:-4]
        iv = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
        sample_dict[iv] = samples.values
        if iv_str != '': ivs.append(int(iv_str))

    obs_samples = sample_dict[frozenset()]
    suffstat = dict(C=np.corrcoef(obs_samples, rowvar=False), n=obs_samples.shape[0])
    return sample_dict, suffstat


def get_sample_dict2():
    perturbations = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'perturbations.npy'))
    perturbation2ix = {p: ix for ix, p in enumerate(perturbations)}
    genes = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'genes.npy'))
    gene2ix = {g: ix for ix, g in enumerate(genes)}
    perturbation_per_cell = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'perturbation_per_cell.npy'))
    total_count_matrix = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'total_count_matrix.npy'))
    total_count_matrix = np.log1p(total_count_matrix)

    # === GET OBSERVATIONAL DATA
    control = 'm_MouseNTC_100_A_67005'
    control_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[control])
    obs_samples = total_count_matrix[:, control_cell_ixs].squeeze().T

    setting_list = []
    for pnum, perturbation in enumerate(perturbations):
        if perturbation != control:
            iv_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[perturbation])
            iv_samples = total_count_matrix[:, iv_cell_ixs].squeeze().T
            target_gene = perturbation[2:-2]
            setting_list.append({'known_interventions': {gene2ix[target_gene]}, 'samples': iv_samples})

    return obs_samples, setting_list


def get_sample_dict2_reduced():
    random.seed(1729)
    np.random.seed(1729)
    perturbations = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'perturbations.npy'))
    perturbation2ix = {p: ix for ix, p in enumerate(perturbations)}
    genes = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'genes.npy'))
    gene2ix = {g: ix for ix, g in enumerate(genes)}
    perturbation_per_cell = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'perturbation_per_cell.npy'))
    total_count_matrix = np.load(os.path.join(DIXIT_FOLDER, 'data2', 'total_count_matrix.npy'))
    total_count_matrix = np.log1p(total_count_matrix)

    # === GET OBSERVATIONAL DATA
    control = 'm_MouseNTC_100_A_67005'
    control_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[control])
    obs_samples = total_count_matrix[:, control_cell_ixs].squeeze().T
    obs_samples = obs_samples[random.sample(list(range(obs_samples.shape[0])), 50)]

    setting_list = []
    for pnum, perturbation in enumerate(perturbations):
        if perturbation != control:
            iv_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[perturbation])
            iv_samples = total_count_matrix[:, iv_cell_ixs].squeeze().T
            iv_samples = iv_samples[random.sample(list(range(iv_samples.shape[0])), 50)]
            target_gene = perturbation[2:-2]
            setting_list.append({'known_interventions': {gene2ix[target_gene]}, 'samples': iv_samples})

    return obs_samples, setting_list




