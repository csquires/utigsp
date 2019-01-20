from real_data_analysis.dixit.dixit_meta import get_sample_dict2
from scipy.stats import ranksums
from collections import defaultdict

SIGNIFICANCE = 0.05
obs_samples, setting_list = get_sample_dict2()

ivs2significant_effects = defaultdict(set)
for setting_num, setting in enumerate(setting_list):
    iv_samples = setting['samples']
    iv_node = list(setting['known_interventions'])[0]
    for col in range(iv_samples.shape[1]):
        if col != iv_node:
            res = ranksums(obs_samples[:, col], iv_samples[:, col])
            print(iv_node, col, res.pvalue)
            if res.pvalue < SIGNIFICANCE:
                ivs2significant_effects[iv_node].add(col)

__all__ = ['ivs2significant_effects']


