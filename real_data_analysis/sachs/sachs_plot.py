from real_data_analysis.sachs.sachs_meta import true_dag, ESTIMATED_FOLDER, nnodes, SACHS_FOLDER
import os
import numpy as np
import pandas as pd
import causaldag as cd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plot_config import ALGS2MARKERS
import seaborn as sns
import re
sns.set()

true_skel = set(map(lambda arc: tuple(sorted(arc)), true_dag.arcs))
true_amat = np.zeros([nnodes, nnodes], dtype=bool)
true_skel_amat = np.zeros([nnodes, nnodes], dtype=bool)
for i, j in true_skel:
    true_skel_amat[i, j] = True
    true_skel_amat[j, i] = True
    true_amat[i, j] = True
npossible_arcs = nnodes*(nnodes-1)
npossible_arcs_skel = npossible_arcs/2

alpha_invariance = 1e-5
# === LOAD DAGS
est_dags_igsp_gauss_ci = []
est_dags_utigsp_gauss_ci = []
est_dags_igsp_hsic = []
est_dags_utigsp_hsic = []
est_dags_gies = []
est_amats_icp = []

igsp_gauss_ci_alphas = []
igsp_hsic_alphas = []
utigsp_gauss_ci_alphas = []
utigsp_hsic_alphas = []
gies_lambdas = []
float_format = '%s'
for file in os.listdir(ESTIMATED_FOLDER):
    amat = np.loadtxt(os.path.join(ESTIMATED_FOLDER, file))
    if file.startswith('icp'):  # not necessarily a DAG
        est_amats_icp.append(amat)
    else:
        dag = cd.DAG.from_amat(amat)
        if file.startswith('utigsp_gauss_ci') and file.endswith('alpha_i=%.2e.txt' % alpha_invariance):
            est_dags_utigsp_gauss_ci.append(dag)
            alpha = float(file.split(',')[0].split('=')[1][:-4])
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
            lam = float(file[len('gies_lambda='):-4])
            gies_lambdas.append(float_format % lam)


# === CREATE DATAFRAMES
def to_df(est_dags, labels=None):
    fps = [len(est_dag.arcs - true_dag.arcs) for est_dag in est_dags]
    tps = [len(true_dag.arcs & est_dag.arcs) for est_dag in est_dags]
    est_skels = [set(map(lambda arc: tuple(sorted(arc)), est_dag.arcs)) for est_dag in est_dags]
    fps_skel = [len(est_skel - true_skel) for est_skel in est_skels]
    tps_skel = [len(est_skel & true_skel) for est_skel in est_skels]
    return pd.DataFrame({'dag': est_dags, 'fp': fps, 'tp': tps, 'fp_skel': fps_skel, 'tp_skel': tps_skel, 'label': labels})


def to_df_icp(est_amats, labels=None):
    fps = [np.sum(np.logical_and(est_amat, np.logical_not(true_amat))) for est_amat in est_amats]
    tps = [np.sum(np.logical_and(est_amat, true_amat)) for est_amat in est_amats]
    est_skel_amats = [(amat + amat.T).astype(bool) for amat in est_amats]
    fps_skel = [np.sum(np.logical_and(est_skel_amat, np.logical_not(true_skel_amat))) for est_skel_amat in est_skel_amats]
    tps_skel = [np.sum(np.logical_and(est_skel_amat, true_skel_amat)) for est_skel_amat in est_skel_amats]
    return pd.DataFrame({'dag': est_amats, 'fp': fps, 'tp': tps, 'fp_skel': fps_skel, 'tp_skel': tps_skel, 'label': labels})


igsp_gauss_ci_df = to_df(est_dags_igsp_gauss_ci, igsp_gauss_ci_alphas)
utigsp_gauss_ci_df = to_df(est_dags_utigsp_gauss_ci, utigsp_gauss_ci_alphas)
igsp_hsic_df = to_df(est_dags_igsp_hsic, igsp_hsic_alphas)
utigsp_hsic_df = to_df(est_dags_utigsp_hsic, utigsp_hsic_alphas)
gies_df = to_df(est_dags_gies, gies_lambdas)
icp_df = to_df_icp(est_amats_icp)

METHOD = 'gauss_ci'
if METHOD == 'gauss_ci':
    igsp_df = igsp_gauss_ci_df
    utigsp_df = utigsp_gauss_ci_df
else:
    igsp_df = igsp_hsic_df
    utigsp_df = utigsp_hsic_df
# === PLOT ROC OF DAG ARCS
plt.clf()
plt.scatter(gies_df.sort_values(by='fp')['fp'], gies_df.sort_values(by='fp')['tp'], label='GIES', marker=ALGS2MARKERS['gies'])
plt.scatter(icp_df.sort_values(by='fp')['fp'], icp_df.sort_values(by='fp')['tp'], label='ICP', marker=ALGS2MARKERS['icp'])
plt.scatter(igsp_df.sort_values(by='fp')['fp'], igsp_df.sort_values(by='fp')['tp'], label='IGSP', marker=ALGS2MARKERS['igsp'])
# for _, row in igsp_hsic_df.iterrows():
#     plt.annotate(row['label'], (row['fp'], row['tp']))
plt.scatter(utigsp_df.sort_values(by='fp')['fp'], utigsp_df.sort_values(by='fp')['tp'], label='UTIGSP', marker=ALGS2MARKERS['utigsp'])
# plt.plot([0, npossible_arcs - len(true_dag.arcs)], [0, len(true_dag.arcs)], color='grey')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
plt.savefig(os.path.join(SACHS_FOLDER, 'figures', 'sachs_roc.png'))

# === PLOT ROC OF SKELETON
plt.clf()
plt.scatter(
    gies_df.sort_values(by='fp_skel')['fp_skel'],
    gies_df.sort_values(by='fp_skel')['tp_skel'],
    label='GIES',
    marker=ALGS2MARKERS['gies']
)
plt.scatter(
    icp_df.sort_values(by='fp_skel')['fp_skel'],
    icp_df.sort_values(by='fp_skel')['tp_skel'],
    label='ICP',
    marker=ALGS2MARKERS['icp']
)
plt.scatter(
    igsp_df.sort_values(by='fp_skel')['fp_skel'],
    igsp_df.sort_values(by='fp_skel')['tp_skel'],
    label='IGSP',
    marker=ALGS2MARKERS['igsp']
)
plt.scatter(
    utigsp_df.sort_values(by='fp_skel')['fp_skel'],
    utigsp_df.sort_values(by='fp_skel')['tp_skel'],
    label='UTIGSP',
    marker=ALGS2MARKERS['utigsp']
)
# for _, row in igsp_hsic_df.iterrows():
#     plt.annotate(row['label'], (row['fp_skel'], row['tp_skel']))
plt.plot([0, npossible_arcs_skel - len(true_skel)], [0, len(true_skel)], color='grey')
plt.yticks(list(range(0, 19, 3)))
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
plt.title('Skeleton')
plt.savefig(os.path.join(SACHS_FOLDER, 'figures', 'sachs_roc_skeleton.png'))

