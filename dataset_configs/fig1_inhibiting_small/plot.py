from dataset_configs.fig1_inhibiting_small.configs import dag_config, sample_config, alg_config, DATASET_NAME
from dataset_configs import evaluate
from config import FIGURES_FOLDER
import os
from plot_config import ALGS2COLORS, LINESTYLES, ALG_HANDLES, create_line_handles

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

true_and_estimated = evaluate.load_true_and_estimated(dag_config, sample_config, alg_config)
shd_array_dict, imec_array_dict = evaluate.get_shd_array(dag_config, sample_config, alg_config, true_and_estimated)
nsettings = 3
shd_array_igsp = shd_array_dict['igsp'].sel(nsettings=nsettings)
shd_array_utigsp = shd_array_dict['utigsp'].sel(nsettings=nsettings)
shd_array_gies = shd_array_dict['gies'].sel(lambda_=50).sel(nsettings=nsettings)
imec_array_igsp = imec_array_dict['igsp'].sel(nsettings=nsettings)
imec_array_utigsp = imec_array_dict['utigsp'].sel(nsettings=nsettings)
imec_array_gies = imec_array_dict['gies'].sel(lambda_=50).sel(nsettings=nsettings)

mean_shd_igsp = shd_array_igsp.mean(dim='dag')
percent_consistent_igsp = (shd_array_igsp == 0).mean(dim='dag')
mean_shd_utigsp = shd_array_utigsp.mean(dim='dag')
percent_consistent_utigsp = (shd_array_utigsp == 0).mean(dim='dag')
mean_shd_gies = shd_array_gies.mean(dim='dag')
percent_consistent_gies = (shd_array_gies == 0).mean(dim='dag')

percent_correct_imec_igsp = imec_array_igsp.mean(dim='dag')
percent_correct_imec_utigsp = imec_array_utigsp.mean(dim='dag')
percent_correct_imec_gies = imec_array_gies.mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, DATASET_NAME), exist_ok=True)
# ==== PLOT MEAN SHD ===
plt.clf()

ntargets_list = mean_shd_igsp.coords['ntargets'].values
handles = [
    *ALG_HANDLES,
    *create_line_handles(ntargets_list)
]
for ntargets, ls in zip(ntargets_list, LINESTYLES):
    plt.plot(mean_shd_igsp.nsamples, mean_shd_igsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['igsp'], linestyle=ls)
    plt.plot(mean_shd_utigsp.nsamples, mean_shd_utigsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['utigsp'], linestyle=ls)
    plt.plot(mean_shd_gies.nsamples, mean_shd_gies.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['gies'], linestyle=ls)
plt.legend(handles=handles)
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
for ntargets, ls in zip(ntargets_list, LINESTYLES):
    plt.plot(percent_consistent_igsp.nsamples, percent_consistent_igsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['igsp'], linestyle=ls)
    plt.plot(percent_consistent_utigsp.nsamples, percent_consistent_utigsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['utigsp'], linestyle=ls)
    plt.plot(percent_consistent_gies.nsamples, percent_consistent_gies.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['gies'], linestyle=ls)
plt.legend(handles=handles)
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Proportion correctly estimated')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'percent_consistent.png'))


# ==== PLOT PERCENT CORRECT I-MEC ===
plt.clf()
for ntargets, ls in zip(ntargets_list, LINESTYLES):
    plt.plot(percent_correct_imec_igsp.nsamples, percent_correct_imec_igsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['igsp'], linestyle=ls)
    plt.plot(percent_correct_imec_utigsp.nsamples, percent_correct_imec_utigsp.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['utigsp'], linestyle=ls)
    plt.plot(percent_correct_imec_gies.nsamples, percent_correct_imec_gies.sel(ntargets=ntargets).squeeze(), color=ALGS2COLORS['gies'], linestyle=ls)
plt.legend(handles=handles)
# plt.xticks(mean_shd_igsp.nsamples)
plt.xlabel('Number of samples')
plt.ylabel('Proportion correct I-MEC')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'percent_correct_imec.png'))
