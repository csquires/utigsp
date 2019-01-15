from dataset_configs.icp_basic.configs import dag_config, sample_config, alg_config, DATASET_NAME
from dataset_configs import evaluate
from config import FIGURES_FOLDER
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

true_and_estimated = evaluate.load_true_and_estimated(dag_config, sample_config, alg_config)
shd_array_dict = evaluate.get_shd_array(dag_config, sample_config, alg_config, true_and_estimated)
shd_array_icp = shd_array_dict['icp']

mean_shd_icp = shd_array_icp.mean(dim='dag')
percent_consistent_icp = (shd_array_icp == 0).mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, DATASET_NAME), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()
for alpha in mean_shd_icp.coords['alpha'].values:
    plt.plot(mean_shd_icp.nsamples, mean_shd_icp.sel(alpha=alpha).squeeze(), label=r'ICP $\alpha_i$=%s' % alpha)
plt.legend()
# plt.xticks(mean_shd_icp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
for alpha in mean_shd_icp.coords['alpha'].values:
    plt.plot(percent_consistent_icp.nsamples, percent_consistent_icp.sel(alpha=alpha).squeeze(), label=r'ICP $\alpha_i$=%s' % alpha)
plt.legend()
# plt.xticks(mean_shd_icp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Proportion correctly estimated')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'percent_consistent.png'))
