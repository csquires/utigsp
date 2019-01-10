from dataset_configs.test.configs import dag_config, sample_config, alg_config
from dataset_configs import evaluate
from config import FIGURES_FOLDER
import os
import utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

true_and_estimated = evaluate.load_true_and_estimated(dag_config, sample_config, alg_config)
shd_array = evaluate.get_shd_array(dag_config, sample_config, alg_config, true_and_estimated)
mean_shd = shd_array.mean(dim='dag')
percent_consistent = (shd_array == 0).mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, 'test'), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()
plt.plot(sample_config.nsamples_list, mean_shd.squeeze())
plt.xticks(sample_config.nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Average SHD')
plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, 'test', 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
plt.plot(sample_config.nsamples_list, percent_consistent.squeeze())
plt.xticks(sample_config.nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Proportion with same skeleton')
plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, 'test', 'percent_consistent.png'))
