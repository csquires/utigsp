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
from simulations.fig1_50.fig1_50_settings import *
from matplotlib.patches import Patch
matplotlib.rc('legend', fontsize=18)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('figure', figsize=(24, 6))

NAME = 'fig1_50'
PLT_FOLDER = os.path.join(PROJECT_FOLDER, 'simulations', 'figures', NAME)
os.makedirs(PLT_FOLDER, exist_ok=True)

# what to plot
UTIGSP = True
IGSP = False
JCIGSP = False
GIES = True
ICP = False

dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, float(nneighbors), ndags)

coords = {
    'nsamples': nsamples_list,
    'num_unknown': [0, 1, 2, 3],
    'dag': list(range(ndags))
}

if GIES:
    shd_array_gies = utils.empty_array(coords)
    imec_array_gies = utils.empty_array(coords)
    shd_icpdag_array_gies = utils.empty_array(coords)
    consistent_array_gies = utils.empty_array(coords)
if ICP:
    shd_array_icp = utils.empty_array(coords)
    imec_array_icp = utils.empty_array(coords)
    shd_icpdag_array_icp = utils.empty_array(coords)
    consistent_array_icp = utils.empty_array(coords)
if IGSP:
    shd_array_igsp = utils.empty_array(coords)
    imec_array_igsp = utils.empty_array(coords)
    shd_icpdag_array_igsp = utils.empty_array(coords)
    consistent_array_igsp = utils.empty_array(coords)
if UTIGSP:
    shd_array_utigsp = utils.empty_array(coords)
    imec_array_utigsp = utils.empty_array(coords)
    shd_icpdag_array_utigsp = utils.empty_array(coords)
    consistent_array_utigsp = utils.empty_array(coords)
    learned_iv_array = utils.empty_array(coords)
    missing_iv_array = utils.empty_array(coords)
    added_iv_array = utils.empty_array(coords)
if JCIGSP:
    shd_array_jcigsp = utils.empty_array(coords)
    imec_array_jcigsp = utils.empty_array(coords)
    shd_icpdag_array_jcigsp = utils.empty_array(coords)
    consistent_array_jcigsp = utils.empty_array(coords)
    # learned_iv_array = utils.empty_array(coords)
    # missing_iv_array = utils.empty_array(coords)
    # added_iv_array = utils.empty_array(coords)

for nsamples, nsettings, (num_known, num_unknown) in itr.product(nsamples_list, nsettings_list, ntargets_list):
    setting_str = f'nsamples={nsamples},num_known={num_known},num_unknown={num_unknown},nsettings={nsettings},intervention={intervention}'
    loc = dict(nsamples=nsamples, num_unknown=num_unknown)

    # === LOAD GIES RESULTS
    if GIES:
        gies_results_folder = os.path.join(
            PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'gies',
            'lambda_=%.2e' % 1
        )
        shd_array_gies.loc[loc] = np.load(os.path.join(gies_results_folder, 'shds.npy'))
        imec_array_gies.loc[loc] = np.load(os.path.join(gies_results_folder, 'imec.npy'))
        shd_icpdag_array_gies.loc[loc] = np.load(os.path.join(gies_results_folder, 'shds_pdag.npy'))
        consistent_array_gies.loc[loc] = np.load(os.path.join(gies_results_folder, 'same_icpdag.npy'))

    # === LOAD ICP RESULTS
    if ICP:
        icp_results_folder = os.path.join(
            PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'icp',
            'alpha=%.2e' % .01
        )
        shd_array_icp.loc[loc] = np.load(os.path.join(icp_results_folder, 'shds.npy'))
        imec_array_icp.loc[loc] = np.load(os.path.join(icp_results_folder, 'imec.npy'))
        shd_icpdag_array_icp.loc[loc] = np.load(os.path.join(icp_results_folder, 'shds_pdag.npy'))
        consistent_array_icp.loc[loc] = np.load(os.path.join(icp_results_folder, 'same_icpdag.npy'))

    # === LOAD IGSP RESULTS
    if IGSP:
        igsp_results_folder = os.path.join(
            PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'igsp',
            'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto'
        )
        shd_array_igsp.loc[loc] = np.load(os.path.join(igsp_results_folder, 'shds.npy'))
        imec_array_igsp.loc[loc] = np.load(os.path.join(igsp_results_folder, 'imec.npy'))
        shd_icpdag_array_igsp.loc[loc] = np.load(os.path.join(igsp_results_folder, 'shds_pdag.npy'))
        consistent_array_igsp.loc[loc] = np.load(os.path.join(igsp_results_folder, 'same_icpdag.npy'))

    # === LOAD UTIGSP RESULTS
    if UTIGSP:
        utigsp_results_folder = os.path.join(
            PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'utigsp',
            'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto'
        )
        if os.path.exists(os.path.join(utigsp_results_folder)):
            shd_array_utigsp.loc[loc] = np.load(os.path.join(utigsp_results_folder, 'shds.npy'))
            imec_array_utigsp.loc[loc] = np.load(os.path.join(utigsp_results_folder, 'imec.npy'))
            shd_icpdag_array_utigsp.loc[loc] = np.load(os.path.join(utigsp_results_folder, 'shds_pdag.npy'))
            consistent_array_utigsp.loc[loc] = np.load(os.path.join(utigsp_results_folder, 'same_icpdag.npy'))
            learned_iv_array.loc[loc] = np.mean(np.load(os.path.join(utigsp_results_folder, 'diff_interventions.npy')), axis=1)
            missing_iv_array.loc[loc] = np.mean(np.load(os.path.join(utigsp_results_folder, 'missing_interventions.npy')), axis=1)
            added_iv_array.loc[loc] = np.mean(np.load(os.path.join(utigsp_results_folder, 'added_interventions.npy')), axis=1)

    # === LOAD JCIGSP RESULTS
    if JCIGSP:
        jcigsp_results_folder = os.path.join(
            PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'jcigsp',
            'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto'
        )
        if os.path.exists(os.path.join(jcigsp_results_folder)):
            shd_array_jcigsp.loc[loc] = np.load(os.path.join(jcigsp_results_folder, 'shds.npy'))
            imec_array_jcigsp.loc[loc] = np.load(os.path.join(jcigsp_results_folder, 'imec.npy'))
            shd_icpdag_array_jcigsp.loc[loc] = np.load(os.path.join(jcigsp_results_folder, 'shds_pdag.npy'))
            consistent_array_jcigsp.loc[loc] = np.load(os.path.join(jcigsp_results_folder, 'same_icpdag.npy'))
            # learned_iv_array.loc[loc] = np.mean(
            #     np.load(os.path.join(jcigsp_results_folder, 'diff_interventions.npy')), axis=1)
            # missing_iv_array.loc[loc] = np.mean(
            #     np.load(os.path.join(jcigsp_results_folder, 'missing_interventions.npy')), axis=1)
            # added_iv_array.loc[loc] = np.mean(
            #     np.load(os.path.join(jcigsp_results_folder, 'added_interventions.npy')), axis=1)

# === CREATE HANDLES
marker_handles = create_marker_handles(map(lambda s: '$\ell=%d$' % s, [0, 1, 2, 3]))
alg_handles = []
if GIES:
    alg_handles.append(Patch(color=ALGS2COLORS['gies'], label='GIES'))
if ICP:
    alg_handles.append(Patch(color=ALGS2COLORS['icp'], label='ICP'))
if IGSP:
    alg_handles.append(Patch(color=ALGS2COLORS['igsp'], label='IGSP'))
if UTIGSP:
    alg_handles.append(Patch(color=ALGS2COLORS['utigsp'], label='UT-IGSP'))
if JCIGSP:
    alg_handles.append(Patch(color=ALGS2COLORS['jcigsp'], label='JCI-GSP'))

# === PLOT SHDS
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    if GIES:
        ax.plot(nsamples_list, shd_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    if ICP:
        ax.plot(nsamples_list, shd_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    if IGSP:
        ax.plot(nsamples_list, shd_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    if UTIGSP:
        ax.plot(nsamples_list, shd_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    if JCIGSP:
        ax.plot(nsamples_list, shd_array_jcigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['jcigsp'])
    ax.set_xlabel('$\ell=%d$' % num_unknown)
# plt.xlabel('Number of samples')
axes[0].set_ylabel('Average SHD')
axes[0].legend(handles=alg_handles, loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'shd.png'))

# === PLOT SHDS OF I-CPDAGS
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    if GIES:
        ax.plot(nsamples_list, shd_icpdag_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    if ICP:
        ax.plot(nsamples_list, shd_icpdag_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    if IGSP:
        ax.plot(nsamples_list, shd_icpdag_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    if UTIGSP:
        ax.plot(nsamples_list, shd_icpdag_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    if JCIGSP:
        ax.plot(nsamples_list, shd_icpdag_array_jcigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['jcigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
# fig.xlabel('Number of samples')
axes[0].set_ylabel('Average SHD')
axes[0].legend(handles=alg_handles, loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'shd-icpdag.png'))

# === PLOT PROPORTIONS CORRECT I-MEC
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    if GIES:
        ax.plot(nsamples_list, imec_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    if ICP:
        ax.plot(nsamples_list, imec_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    if IGSP:
        ax.plot(nsamples_list, imec_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    if UTIGSP:
        ax.plot(nsamples_list, imec_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    if JCIGSP:
        ax.plot(nsamples_list, imec_array_jcigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['jcigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
axes[0].set_ylabel('Proportion in the true I-MEC')
axes[-1].legend(handles=alg_handles, loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'correct-imec.png'))

# === PLOT PROPORTION SAME ICPDAG
plt.clf()
fig, axes = plt.subplots(1, 4, sharey=True)
for num_unknown, ax in zip([0, 1, 2, 3], axes):
    if GIES:
        ax.plot(nsamples_list, consistent_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'])
    if ICP:
        ax.plot(nsamples_list, consistent_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'])
    if IGSP:
        ax.plot(nsamples_list, consistent_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'])
    if UTIGSP:
        ax.plot(nsamples_list, consistent_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'])
    if JCIGSP:
        ax.plot(nsamples_list, consistent_array_jcigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['jcigsp'])
    ax.set_xticks(nsamples_list)
    ax.set_xlabel('$\ell=%d$' % num_unknown)
axes[0].set_ylabel('Proportion consistently estimated I-MECs')
axes[-1].legend(handles=alg_handles, loc='upper center')
plt.tight_layout()
plt.savefig(os.path.join(PLT_FOLDER, 'consistent-icpdag.png'))


# === PLOT DIFFERENCE IN NUMBER OF INTERVENTION TARGETS RECOVERED
plt.clf()
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    axes[0].plot(nsamples_list, missing_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    axes[1].plot(nsamples_list, added_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
    # plt.plot(nsamples_list, missing_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
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
    # plt.plot(nsamples_list, learned_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    plt.plot(nsamples_list, missing_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Average number of false negatives')
plt.legend(handles=[
    *reversed(marker_handles),
])
plt.savefig(os.path.join(PLT_FOLDER, 'missing-targets.png'))

plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    # plt.plot(nsamples_list, learned_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    # plt.plot(nsamples_list, missing_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    plt.plot(nsamples_list, added_iv_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Average number of false positives')
plt.legend(handles=[
    *reversed(marker_handles),
])
plt.savefig(os.path.join(PLT_FOLDER, 'added-targets.png'))

