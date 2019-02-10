import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from config import PROJECT_FOLDER
import os
import numpy as np
import itertools as itr
import utils
from plot_config import ALGS2COLORS, MARKERS, create_marker_handles, ALG_HANDLES
from matplotlib.patches import Patch
matplotlib.rc('legend', fontsize=18)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('figure', figsize=(24, 6))

NAME = 'fig1_soft'
PLT_FOLDER = os.path.join(PROJECT_FOLDER, 'simulations', 'figures', NAME)
os.makedirs(PLT_FOLDER, exist_ok=True)

nnodes = 10
nneighbors = 1.5
ndags = 100
dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)

nsamples_list = [100, 200, 300, 400, 500]
nsettings_list = [5]
ntargets_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
intervention = 'soft1'

coords = {
    'nsamples': nsamples_list,
    'num_unknown': [0, 1, 2, 3],
    'dag': list(range(ndags))
}
shd_array_gies = utils.empty_array(coords)
imec_array_gies = utils.empty_array(coords)
shd_icpdag_array_gies = utils.empty_array(coords)
consistent_array_gies = utils.empty_array(coords)

shd_array_icp = utils.empty_array(coords)
imec_array_icp = utils.empty_array(coords)
shd_icpdag_array_icp = utils.empty_array(coords)
consistent_array_icp = utils.empty_array(coords)

shd_array_igsp = utils.empty_array(coords)
imec_array_igsp = utils.empty_array(coords)
shd_icpdag_array_igsp = utils.empty_array(coords)
consistent_array_igsp = utils.empty_array(coords)

shd_array_utigsp = utils.empty_array(coords)
imec_array_utigsp = utils.empty_array(coords)
shd_icpdag_array_utigsp = utils.empty_array(coords)
consistent_array_utigsp = utils.empty_array(coords)
learned_intervention_array = utils.empty_array(coords)
missing_intervention_array = utils.empty_array(coords)
added_intervention_array = utils.empty_array(coords)

for nsamples, nsettings, (num_known, num_unknown) in itr.product(nsamples_list, nsettings_list, ntargets_list):
    setting_str = f'nsamples={nsamples},num_known={num_known},num_unknown={num_unknown},nsettings={nsettings},intervention={intervention}'
    loc = dict(nsamples=nsamples, num_unknown=num_unknown)

    # === LOAD GIES RESULTS
    gies_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'gies', 'lambda_=%.2e' % 1)
    shd_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'shds.txt'))
    imec_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'imec.txt'))
    shd_icpdag_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'shds_pdag.txt'))
    consistent_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'same_icpdag.txt'))

    # # === LOAD ICP RESULTS
    # icp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'icp', 'alpha=%.2e' % .01)
    # shd_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'shds.txt'))
    # imec_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'imec.txt'))
    # shd_icpdag_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'shds_pdag.txt'))
    # consistent_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'same_icpdag.txt'))

    # === LOAD IGSP RESULTS
    igsp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'igsp', 'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto')
    shd_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'shds.txt'))
    imec_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'imec.txt'))
    shd_icpdag_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'shds_pdag.txt'))
    consistent_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'same_icpdag.txt'))

    # === LOAD UTIGSP RESULTS
    utigsp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'utigsp', 'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto')
    if os.path.exists(os.path.join(utigsp_results_folder)):
        shd_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'shds.txt'))
        imec_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'imec.txt'))
        shd_icpdag_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'shds_pdag.txt'))
        consistent_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'same_icpdag.txt'))
        learned_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'diff_interventions.txt')), axis=1)
        missing_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'missing_interventions.txt')), axis=1)
        added_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'added_interventions.txt')), axis=1)

# === CREATE HANDLES
marker_handles = create_marker_handles(map(lambda s: '$\ell=%d$' % s, [0, 1, 2, 3]))

# === PLOT SHDS
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    ax.plot(nsamples_list, shd_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    ax.plot(nsamples_list, shd_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    ax.plot(nsamples_list, shd_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    ax.plot(nsamples_list, shd_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    ax.set_xlabel('$\ell=%d$' % num_unknown)
# plt.xlabel('Number of samples')
axes[0].set_ylabel('Average SHD')
axes[0].legend(handles=[
    Patch(color=ALGS2COLORS['gies'], label='GIES'),
    Patch(color=ALGS2COLORS['icp'], label='ICP'),
    Patch(color=ALGS2COLORS['igsp'], label='IGSP'),
    Patch(color=ALGS2COLORS['utigsp'], label='UT-IGSP'),
], loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'shd.png'))

# === PLOT SHDS OF I-CPDAGS
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    ax.plot(nsamples_list, shd_icpdag_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    ax.plot(nsamples_list, shd_icpdag_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    ax.plot(nsamples_list, shd_icpdag_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    ax.plot(nsamples_list, shd_icpdag_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
# fig.xlabel('Number of samples')
axes[0].set_ylabel('Average SHD')
axes[0].legend(handles=[
    Patch(color=ALGS2COLORS['gies'], label='GIES'),
    Patch(color=ALGS2COLORS['icp'], label='ICP'),
    Patch(color=ALGS2COLORS['igsp'], label='IGSP'),
    Patch(color=ALGS2COLORS['utigsp'], label='UT-IGSP'),
], loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'shd-icpdag.png'))

# === PLOT PROPORTIONS CORRECT I-MEC
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    ax.plot(nsamples_list, imec_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    ax.plot(nsamples_list, imec_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    ax.plot(nsamples_list, imec_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    ax.plot(nsamples_list, imec_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
axes[0].set_ylabel('Proportion in the true I-MEC')
axes[-1].legend(handles=[
    Patch(color=ALGS2COLORS['gies'], label='GIES'),
    Patch(color=ALGS2COLORS['icp'], label='ICP'),
    Patch(color=ALGS2COLORS['igsp'], label='IGSP'),
    Patch(color=ALGS2COLORS['utigsp'], label='UT-IGSP'),
], loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'correct-imec.png'))

# === PLOT PROPORTION SAME ICPDAG
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    ax.plot(nsamples_list, consistent_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    ax.plot(nsamples_list, consistent_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    ax.plot(nsamples_list, consistent_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    ax.plot(nsamples_list, consistent_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
axes[0].set_ylabel('Proportion consistently estimated I-MECs')
axes[-1].legend(handles=[
    Patch(color=ALGS2COLORS['gies'], label='GIES'),
    Patch(color=ALGS2COLORS['icp'], label='ICP'),
    Patch(color=ALGS2COLORS['igsp'], label='IGSP'),
    Patch(color=ALGS2COLORS['utigsp'], label='UT-IGSP'),
], loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'consistent-icpdag.png'))


# === PLOT DIFFERENCE IN NUMBER OF INTERVENTION TARGETS RECOVERED
plt.clf()
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    axes[0].plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    axes[1].plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
    # plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
axes[0].set_xticks(nsamples_list)
axes[0].set_xlabel('Number of samples')
axes[1].set_xticks(nsamples_list)
axes[1].set_xlabel('Number of samples')
axes[0].set_ylabel('Average # of false negatives')
axes[1].set_ylabel('Average # of false positives')
axes[0].yaxis.set_label_position('right')
axes[1].yaxis.set_label_position('right')
axes[0].legend(handles=[
    *reversed(marker_handles),
    # Patch(color='k', label='Both'),
    # Patch(color='r', label='Missing'),
    # Patch(color='b', label='Added')
], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'recovered-targets.png'))

plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    # plt.plot(nsamples_list, learned_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Average number of false negatives')
plt.legend(handles=[
    *reversed(marker_handles),
])
plt.savefig(os.path.join(PLT_FOLDER, 'missing-targets.png'))

plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    # plt.plot(nsamples_list, learned_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    # plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Average number of false positives')
plt.legend(handles=[
    *reversed(marker_handles),
])
plt.savefig(os.path.join(PLT_FOLDER, 'added-targets.png'))

