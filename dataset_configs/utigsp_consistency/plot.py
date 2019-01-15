from dataset_configs.utigsp_consistency.configs import dag_config, sample_config, alg_config, DATASET_NAME
from dataset_configs import evaluate
from config import FIGURES_FOLDER
from plot_config import ALGS2COLORS, LINESTYLES
import os

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set()

true_and_estimated = evaluate.load_true_and_estimated(dag_config, sample_config, alg_config)
shd_array_dict = evaluate.get_shd_array(dag_config, sample_config, alg_config, true_and_estimated)
shd_array_utigsp = shd_array_dict['utigsp']

mean_shd_utigsp = shd_array_utigsp.mean(dim='dag')
percent_consistent_utigsp = (shd_array_utigsp == 0).mean(dim='dag')

os.makedirs(os.path.join(FIGURES_FOLDER, DATASET_NAME), exist_ok=True)

# === DEFINE LEGEND HANDLES
handles = [
    mlines.Line2D([], [], color=ALGS2COLORS['utigsp'], label='UTIGSP'),
]
# ==== PLOT MEAN SHD ===
plt.clf()
plt.plot(mean_shd_utigsp.nsamples, mean_shd_utigsp.squeeze(), label='UTIGSP', color=ALGS2COLORS['utigsp'])
plt.legend(handles=handles)
plt.xlabel('Number of samples')
plt.ylabel('Average SHD')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'mean_shd.png'))

# ==== PLOT PERCENT CONSISTENT ===
plt.clf()
plt.plot(percent_consistent_utigsp.nsamples, percent_consistent_utigsp.squeeze(), label='UTIGSP', color=ALGS2COLORS['utigsp'])
plt.legend(handles=handles)
plt.xlabel('Number of samples')
plt.ylabel('Proportion correctly estimated')
# plt.title(utils.make_title(dag_config, sample_config, alg_config, nsamples=False))
plt.savefig(os.path.join(FIGURES_FOLDER, DATASET_NAME, 'percent_consistent.png'))
