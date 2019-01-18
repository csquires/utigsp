from real_data_analysis.sachs.sachs_meta import true_dag, ESTIMATED_FOLDER, nnodes, SACHS_FOLDER
import os
import numpy as np
import pandas as pd
import causaldag as cd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plot_config import ALGS2COLORS
import seaborn as sns
import re
sns.set()

true_skel = set(map(lambda arc: tuple(sorted(arc)), true_dag.arcs))
npossible_arcs = nnodes*(nnodes-1)
npossible_arcs_skel = npossible_arcs/2

alpha_invariance = 1e-5
# === LOAD DAGS
est_dags_igsp_gauss_ci = []
est_dags_utigsp_gauss_ci = []
est_dags_igsp_hsic = []
est_dags_utigsp_hsic = []
est_dags_gies = []

igsp_gauss_ci_alphas = []
igsp_hsic_alphas = []
utigsp_gauss_ci_alphas = []
utigsp_hsic_alphas = []
float_format = '%s'
for file in os.listdir(ESTIMATED_FOLDER):
    dag = cd.DAG.from_amat(np.loadtxt(os.path.join(ESTIMATED_FOLDER, file)))
    if file.startswith('utigsp_gauss_ci') and file.endswith('alpha_i=%.2e.txt' % alpha_invariance):
        est_dags_utigsp_gauss_ci.append(dag)
        alpha = float(re.search('alpha=(\S+),', file).group()[6:-1])  # sorry this is messy
        utigsp_gauss_ci_alphas.append(float_format % alpha)
    if file.startswith('igsp_gauss_ci') and file.endswith('alpha_i=%.2e.txt' % alpha_invariance):
        est_dags_igsp_gauss_ci.append(dag)
        alpha = float(re.search('alpha=(\S+),', file).group()[6:-1])  # sorry this is messy
        igsp_gauss_ci_alphas.append(float_format % alpha)
    if file.startswith('utigsp_hsic') and file.endswith('alpha_i=%.2e.txt' % alpha_invariance):
        est_dags_utigsp_hsic.append(dag)
        alpha = float(re.search('alpha=(\S+),', file).group()[6:-1])  # sorry this is messy
        utigsp_hsic_alphas.append(float_format % alpha)
    if file.startswith('igsp_hsic') and file.endswith('alpha_i=%.2e.txt' % alpha_invariance):
        est_dags_igsp_hsic.append(dag)
        alpha = float(re.search('alpha=(\S+),', file).group()[6:-1])  # sorry this is messy
        igsp_hsic_alphas.append(float_format % alpha)
    if file.startswith('gies'):
        est_dags_gies.append(dag)


# === CREATE DATAFRAMES
def to_df(est_dags, labels=None):
    fps = [len(est_dag.arcs - true_dag.arcs) for est_dag in est_dags]
    tps = [len(true_dag.arcs & est_dag.arcs) for est_dag in est_dags]
    est_skels = [set(map(lambda arc: tuple(sorted(arc)), est_dag.arcs)) for est_dag in est_dags]
    fps_skel = [len(est_skel - true_skel) for est_skel in est_skels]
    tps_skel = [len(est_skel & true_skel) for est_skel in est_skels]
    return pd.DataFrame({'dag': est_dags, 'fp': fps, 'tp': tps, 'fp_skel': fps_skel, 'tp_skel': tps_skel, 'label': labels})


igsp_gauss_ci_df = to_df(est_dags_igsp_gauss_ci, igsp_gauss_ci_alphas)
utigsp_gauss_ci_df = to_df(est_dags_utigsp_gauss_ci, utigsp_gauss_ci_alphas)
igsp_hsic_df = to_df(est_dags_igsp_hsic, igsp_hsic_alphas)
utigsp_hsic_df = to_df(est_dags_utigsp_hsic, utigsp_hsic_alphas)
gies_df = to_df(est_dags_gies)

# === PLOT ROC OF DAG ARCS
plt.clf()
plt.plot(gies_df.sort_values(by='fp')['fp'], gies_df.sort_values(by='fp')['tp'], label='GIES', color=ALGS2COLORS['gies'])
plt.plot(igsp_hsic_df.sort_values(by='fp')['fp'], igsp_hsic_df.sort_values(by='fp')['tp'], label='IGSP', color=ALGS2COLORS['igsp'])
for _, row in igsp_hsic_df.iterrows():
    plt.annotate(row['label'], (row['fp'], row['tp']))
plt.plot(utigsp_hsic_df.sort_values(by='fp')['fp'], utigsp_hsic_df.sort_values(by='fp')['tp'], label='UTIGSP', color=ALGS2COLORS['utigsp'])
plt.plot([0, npossible_arcs - len(true_dag.arcs)], [0, len(true_dag.arcs)], color='grey')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
plt.savefig(os.path.join(SACHS_FOLDER, 'figures', 'roc.png'))

# === PLOT ROC OF SKELETON
plt.clf()
plt.plot(
    gies_df.sort_values(by='fp_skel')['fp_skel'],
    gies_df.sort_values(by='fp_skel')['tp_skel'],
    label='GIES',
    color=ALGS2COLORS['gies']
)
plt.plot(
    igsp_hsic_df.sort_values(by='fp_skel')['fp_skel'],
    igsp_hsic_df.sort_values(by='fp_skel')['tp_skel'],
    label='IGSP',
    color=ALGS2COLORS['igsp']
)
plt.plot(
    utigsp_hsic_df.sort_values(by='fp_skel')['fp_skel'],
    utigsp_hsic_df.sort_values(by='fp_skel')['tp_skel'],
    label='UTIGSP',
    color=ALGS2COLORS['utigsp']
)
for _, row in igsp_hsic_df.iterrows():
    plt.annotate(row['label'], (row['fp_skel'], row['tp_skel']))
plt.plot([0, npossible_arcs_skel - len(true_skel)], [0, len(true_skel)], color='grey')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
plt.title('Skeleton')
plt.savefig(os.path.join(SACHS_FOLDER, 'figures', 'roc_skeleton.png'))
