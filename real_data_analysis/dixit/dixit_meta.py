import os
from config import PROJECT_FOLDER
import numpy as np
import pandas as pd

DIXIT_FOLDER = os.path.join(PROJECT_FOLDER, 'real_data_analysis', 'dixit')
DIXIT_DATA_FOLDER = os.path.join(DIXIT_FOLDER, 'data')
ESTIMATED_FOLDER = os.path.join(DIXIT_FOLDER, 'estimated')
FIGURES_FOLDER = os.path.join(DIXIT_FOLDER, 'figures')
nnodes = 24


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

