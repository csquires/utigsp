from dataset_configs.test.configs import dag_config, sample_config, alg_config
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
shd_array_igsp = shd_array_dict['igsp'].sel(nsettings=5)
shd_array_utigsp = shd_array_dict['utigsp'].sel(nsettings=5)
shd_array_gies = shd_array_dict['gies'].sel(nsettings=5)

mean_shd_igsp = shd_array_igsp.mean(dim='dag')
percent_consistent_igsp = (shd_array_igsp == 0).mean(dim='dag')
mean_shd_utigsp = shd_array_utigsp.mean(dim='dag')
percent_consistent_utigsp = (shd_array_utigsp == 0).mean(dim='dag')
mean_shd_gies = shd_array_gies.mean(dim='dag')
percent_consistent_gies = (shd_array_gies == 0).mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, 'test'), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()
for lambda_ in mean_shd_gies.coords['lambda_'].values:
    plt.plot(mean_shd_gies.nsamples, mean_shd_gies.sel(lambda_=lambda_).squeeze(), label='GIES $\lambda$=%s' % lambda_)
plt.plot(mean_shd_igsp.nsamples, mean_shd_igsp.squeeze(), label='IGSP')
plt.plot(mean_shd_utigsp.nsamples, mean_shd_utigsp.squeeze(), label='UTIGSP')
plt.legend()
plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, 'test', 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
for lambda_ in mean_shd_gies.coords['lambda_'].values:
    plt.plot(percent_consistent_gies.nsamples, percent_consistent_gies.sel(lambda_=lambda_).squeeze(), label='GIES $\lambda$=%s' % lambda_)
plt.plot(percent_consistent_igsp.nsamples, percent_consistent_igsp.squeeze(), label='IGSP')
plt.plot(percent_consistent_utigsp.nsamples, percent_consistent_utigsp.squeeze(), label='UTIGSP')
plt.legend()
plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Proportion correctly estimated')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, 'test', 'percent_consistent.png'))
