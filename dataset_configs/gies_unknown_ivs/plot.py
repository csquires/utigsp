from dataset_configs.gies_unknown_ivs.configs import dag_config, sample_config, alg_config, DATASET_NAME
from dataset_configs import evaluate
from config import FIGURES_FOLDER
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
dataset_name = DATASET_NAME

true_and_estimated = evaluate.load_true_and_estimated(dag_config, sample_config, alg_config)
shd_array_dict = evaluate.get_shd_array(dag_config, sample_config, alg_config, true_and_estimated)
shd_array_gies = shd_array_dict['gies'].sel(nsettings=2, nsamples=500)

mean_shd_gies = shd_array_gies.mean(dim='dag')
percent_consistent_gies = (shd_array_gies == 0).mean(dim='dag')

n_unknown = [0, 1, 2, 3]
ntargets_strs = ['1,%d' % i for i in n_unknown]
os.makedirs(os.path.join(FIGURES_FOLDER, dataset_name), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()
for lambda_ in mean_shd_gies.coords['lambda_'].values:
    plt.plot(n_unknown, mean_shd_gies.sel(lambda_=lambda_, ntargets=ntargets_strs).squeeze(), label='GIES $\lambda$=%s' % lambda_)
plt.legend()
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of unknown interventions')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, dataset_name, 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
for lambda_ in mean_shd_gies.coords['lambda_'].values:
    plt.plot(n_unknown, percent_consistent_gies.sel(lambda_=lambda_, ntargets=ntargets_strs).squeeze(), label='GIES $\lambda$=%s' % lambda_)
plt.legend()
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of unknown interventions')
plt.ylabel('Proportion correctly estimated')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, dataset_name, 'percent_consistent.png'))
