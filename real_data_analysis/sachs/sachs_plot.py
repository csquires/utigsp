from real_data_analysis.sachs.sachs_meta import true_dag, ESTIMATED_FOLDER
import os
import numpy as np
import causaldag as cd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plot_config import ALGS2COLORS

# === LOAD DAGS
est_dags_igsp_gauss_ci = []
est_dags_utigsp_gauss_ci = []
est_dags_igsp_hsic = []
est_dags_utigsp_hsic = []
est_dags_gies = []
for file in os.listdir(ESTIMATED_FOLDER):
    dag = cd.DAG.from_amat(np.loadtxt(os.path.join(ESTIMATED_FOLDER, file)))
    if file.startswith('utigsp_gauss_ci'):
        est_dags_utigsp_gauss_ci.append(dag)
    if file.startswith('igsp_gauss_ci'):
        est_dags_igsp_gauss_ci.append(dag)
    if file.startswith('utigsp_hsic'):
        est_dags_utigsp_hsic.append(dag)
    if file.startswith('igsp_hsic'):
        est_dags_igsp_hsic.append(dag)
    if file.startswith('gies'):
        est_dags_gies.append(dag)

# === GET TUPLES OF FALSE POSITIVES AND TRUE POSITIVES
fp_tp_igsp_gauss_ci = [(len(est_dag.arcs - true_dag.arcs), len(true_dag.arcs & est_dag.arcs)) for est_dag in est_dags_igsp_gauss_ci]
fp_tp_utigsp_gauss_ci = [(len(est_dag.arcs - true_dag.arcs), len(true_dag.arcs & est_dag.arcs)) for est_dag in est_dags_utigsp_gauss_ci]
fp_tp_igsp_hsic = [(len(est_dag.arcs - true_dag.arcs), len(true_dag.arcs & est_dag.arcs)) for est_dag in est_dags_igsp_hsic]
fp_tp_utigsp_hsic = [(len(est_dag.arcs - true_dag.arcs), len(true_dag.arcs & est_dag.arcs)) for est_dag in est_dags_utigsp_hsic]
fp_tp_gies = [(len(est_dag.arcs - true_dag.arcs), len(true_dag.arcs & est_dag.arcs)) for est_dag in est_dags_gies]

fp_tp_igsp_gauss_ci = list(sorted(fp_tp_igsp_gauss_ci))
fp_tp_utigsp_gauss_ci = list(sorted(fp_tp_utigsp_gauss_ci))
fp_tp_igsp_hsic = list(sorted(fp_tp_igsp_hsic))
fp_tp_utigsp_hsic = list(sorted(fp_tp_utigsp_hsic))
fp_tp_gies = list(sorted(fp_tp_gies))

# === PLOT RESULTS
plt.plot(*zip(*fp_tp_gies), label='GIES', color=ALGS2COLORS['gies'])
plt.plot(*zip(*fp_tp_igsp_gauss_ci), label='IGSP', color=ALGS2COLORS['igsp'])
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
# plt.plot(fp_tp_gies, label='GIES')

