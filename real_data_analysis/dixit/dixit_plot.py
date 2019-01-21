from real_data_analysis.dixit.create_significant_effect_list2 import ivs2significant_effects, pvalues
from real_data_analysis.dixit.dixit_meta import nnodes, ESTIMATED_FOLDER, DIXIT_FOLDER
from plot_config import ALGS2MARKERS
import causaldag as cd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

npossible_effects = nnodes*(nnodes-1)
npositives = sum(len(effects) for iv_nodes, effects in ivs2significant_effects.items())

# === GET NUMBER OF DIFFERENT SETTING
folder1 = os.path.join(ESTIMATED_FOLDER, 'exclude_0')
n_gies = sum(file.startswith('gies') for file in os.listdir(folder1))
n_igsp = sum(file.startswith('igsp') for file in os.listdir(folder1))
n_utigsp = sum(file.startswith('utigsp') for file in os.listdir(folder1))

# === TOTAL TRUE AND FALSE POSITIVES ACROSS ALL INTERVENTIONS
total_tp_gies, total_fp_gies = np.zeros(n_gies), np.zeros(n_gies)
total_tp_igsp, total_fp_igsp = np.zeros(n_igsp), np.zeros(n_igsp)
total_tp_utigsp, total_fp_utigsp = np.zeros(n_utigsp), np.zeros(n_utigsp)

# === GO THROUGH EACH INTERVENTION TO FIND TRUE AND FALSE POSITIVES
for excluded_node in range(24):
    folder = os.path.join(ESTIMATED_FOLDER, 'exclude_%d' % excluded_node)

    def get_tp_fp(alg):
        est_dags = [
            cd.DAG.from_amat(np.loadtxt(os.path.join(folder, file)))
            for file in os.listdir(folder)
            if file.startswith(alg)
        ]
        # print(os.listdir(folder))
        est_children_by_dag = [d.children_of(excluded_node) for d in est_dags]
        # print(est_children_by_dag)
        acceptable_children = ivs2significant_effects[excluded_node]
        # print(acceptable_children)
        true_positives = [acceptable_children & est_children for est_children in est_children_by_dag]
        false_positives = [est_children - acceptable_children for est_children in est_children_by_dag]
        return [len(tp) for tp in true_positives], [len(fp) for fp in false_positives]

    tp_gies, fp_gies = get_tp_fp('gies')
    total_tp_gies += tp_gies
    total_fp_gies += fp_gies


# === SORT IN ASCENDING ORDER
def sort_fp_tp(fps, tps):
    sort_ixs = np.argsort(fps)
    return fps[sort_ixs], tps[sort_ixs]


total_fp_gies, total_tp_gies = sort_fp_tp(total_fp_gies, total_tp_gies)

# === PLOT
plt.clf()
plt.scatter(total_fp_gies, total_tp_gies, marker=ALGS2MARKERS['gies'], label='GIES')
plt.plot([0, npossible_effects-npositives], [0, npositives], color='grey')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend()
plt.savefig(os.path.join(DIXIT_FOLDER, 'figures', 'roc.png'))


# === PLOT PVALUES
plt.clf()
plt.hist(pvalues, bins=100)
plt.xlabel('p-value')
plt.ylabel('Count')
plt.savefig(os.path.join(DIXIT_FOLDER, 'figures', 'pvalue-hist.png'))
