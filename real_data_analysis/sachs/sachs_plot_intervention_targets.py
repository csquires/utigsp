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
import json

intervention_targets = [{1}, {3}, {4}, {6}, {8}]
npositives = 5
npossible = 5*11

fp_tp_list = []
for file in os.listdir(ESTIMATED_FOLDER):
    if file.startswith('learned_interventions') and 'alpha=3.00e-01' in file:
        learned_interventions = json.load(open(os.path.join(ESTIMATED_FOLDER, file)))
        false_positives = [set(iv_nodes) - true_iv_nodes for iv_nodes, true_iv_nodes in zip(learned_interventions, intervention_targets)]
        true_positives = [set(iv_nodes) & true_iv_nodes for iv_nodes, true_iv_nodes in zip(learned_interventions, intervention_targets)]
        fp_tp_list.append((sum(map(len, false_positives)), sum(map(len, true_positives))))
fp_tp_list = sorted(fp_tp_list)

plt.clf()
plt.plot(*zip(*fp_tp_list), label='UT-IGSP')
plt.plot([0, npossible-npositives], [0, npositives], color='g')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.savefig(os.path.join(SACHS_FOLDER, 'figures', 'sachs_target_recovery.png'))



