from scipy.stats import ranksums
import os
import pandas as pd
from real_data_analysis.dixit.dixit_meta import DIXIT_DATA_FOLDER
from collections import defaultdict
import numpy as np

SIGNIFICANCE = 0.05
sample_dict = {}
ivs = []
for file in os.listdir(DIXIT_DATA_FOLDER):
    samples = pd.read_csv(os.path.join(DIXIT_DATA_FOLDER, file), sep=',')
    iv_str = file.split('=')[1][:-4]
    iv = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
    sample_dict[iv] = samples.values
    if iv_str != '': ivs.append(int(iv_str))

ivs2significant_effects = defaultdict(set)
obs_samples = sample_dict[frozenset()]
for iv_nodes, iv_samples in sample_dict.items():
    if iv_nodes != frozenset():
        for col in range(iv_samples.shape[1]):
            if col not in iv_nodes:
                res = ranksums(iv_samples[:, col], obs_samples[:, col])
                print(iv_nodes, np.log(res.pvalue))
                if res.pvalue < SIGNIFICANCE:
                    ivs2significant_effects[iv_nodes].add(col)

__all__ = ['ivs2significant_effects']



