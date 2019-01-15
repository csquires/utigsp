from dataset_configs.igsp_v_gies.configs import dag_config, sample_config, alg_config, DATASET_NAME
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
shd_array_igsp = shd_array_dict['igsp']
shd_array_gies = shd_array_dict['gies']

mean_shd_igsp = shd_array_igsp.mean(dim='dag')
percent_consistent_igsp = (shd_array_igsp == 0).mean(dim='dag')
mean_shd_gies = shd_array_gies.mean(dim='dag')
percent_consistent_gies = (shd_array_gies == 0).mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, DATASET_NAME), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()
for alpha in mean_shd_igsp.coords['alpha_invariant'].values:
    plt.plot(mean_shd_igsp.nsamples, mean_shd_igsp.sel(alpha_invariant=alpha).squeeze(), label=r'IGSP $\alpha=%s$' % alpha)
for lambda_ in mean_shd_gies.coords['lambda_'].values:
    plt.plot(mean_shd_gies.nsamples, mean_shd_gies.sel(lambda_=lambda_).squeeze(), label='GIES $\lambda$=%s' % lambda_)
plt.legend()
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of intervention settings')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
# plt.clf()
# plt.plot(percent_consistent_igsp.nsamples, percent_consistent_igsp.squeeze(), label='IGSP')
# for lambda_ in mean_shd_gies.coords['lambda_'].values:
#     plt.plot(percent_consistent_gies.nsamples, percent_consistent_gies.sel(lambda_=lambda_).squeeze(), label='GIES $\lambda$=%s' % lambda_)
# plt.legend()
# # plt.xticks(mean_shd_igsp.nsamples)
# plt.xlabel('Number of intervention settings')
# plt.ylabel('Proportion correctly estimated')
# # plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
# plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'percent_consistent.png'))
