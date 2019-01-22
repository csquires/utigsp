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

NAME = 'fig5'
PLT_FOLDER = os.path.join(PROJECT_FOLDER, 'simulations', 'figures', NAME)
os.makedirs(PLT_FOLDER, exist_ok=True)

nnodes = 8
nneighbors = 1.5
ndags = 100
dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)

nsamples_list = [100, 300, 500]
nsettings_list = [8]
ntargets_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
intervention = 'perfect1'

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

    # === LOAD IGSP RESULTS
    igsp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'igsp', 'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto')
    igsp_exists = os.path.exists(igsp_results_folder)
    shd_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'shds.txt')) if igsp_exists else None
    imec_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'imec.txt')) if igsp_exists else None
    shd_icpdag_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'shds_pdag.txt')) if igsp_exists else None
    consistent_array_igsp.loc[loc] = np.loadtxt(os.path.join(igsp_results_folder, 'same_icpdag.txt')) if igsp_exists else None

    # === LOAD UTIGSP RESULTS
    utigsp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'utigsp', 'nruns=10,depth=4,alpha=1.00e-05,alpha_invariant=1.00e-05,pool=auto')
    utigsp_exists = os.path.exists(utigsp_results_folder)
    shd_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'shds.txt')) if utigsp_exists else None
    imec_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'imec.txt')) if utigsp_exists else None
    shd_icpdag_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'shds_pdag.txt')) if utigsp_exists else None
    consistent_array_utigsp.loc[loc] = np.loadtxt(os.path.join(utigsp_results_folder, 'same_icpdag.txt')) if utigsp_exists else None
    learned_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'diff_interventions.txt')), axis=1)
    # missing_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'missing_interventions.txt')), axis=1)
    # added_intervention_array.loc[loc] = np.mean(np.loadtxt(os.path.join(utigsp_results_folder, 'added_interventions.txt')), axis=1)

# === CREATE HANDLES
marker_handles = create_marker_handles([0, 1, 2, 3])

# === PLOT SHDS
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, shd_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    # plt.plot(nsamples_list, shd_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
    plt.plot(nsamples_list, shd_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'], marker=marker)
    plt.plot(nsamples_list, shd_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('SHD')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.ylim([0, 8])
plt.savefig(os.path.join(PLT_FOLDER, 'shd.png'))

# === PLOT SHDS OF I-CPDAGS
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, shd_icpdag_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    # plt.plot(nsamples_list, shd_icpdag_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
    plt.plot(nsamples_list, shd_icpdag_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'], marker=marker)
    plt.plot(nsamples_list, shd_icpdag_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('SHD')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.ylim([0, 8])
plt.savefig(os.path.join(PLT_FOLDER, 'shd_icpdag.png'))

# === PLOT PROPORTIONS CORRECT I-MEC
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, imec_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    # plt.plot(nsamples_list, imec_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
    plt.plot(nsamples_list, imec_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'], marker=marker)
    plt.plot(nsamples_list, imec_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Proportion in the true I-MEC')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.savefig(os.path.join(PLT_FOLDER, 'correct_imec.png'))

# === PLOT PROPORTION SAME ICPDAG
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, consistent_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    # plt.plot(nsamples_list, consistent_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
    plt.plot(nsamples_list, consistent_array_igsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['igsp'], marker=marker)
    plt.plot(nsamples_list, consistent_array_utigsp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['utigsp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Proportion consistently estimated I-CPDAGs')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.savefig(os.path.join(PLT_FOLDER, 'consistent_icpdag.png'))

# === PLOT DIFFERENCE IN NUMBER OF INTERVENTION TARGETS RECOVERED
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, learned_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    # plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Mean symmetric difference in recovered targets')
plt.legend(handles=[
    *marker_handles,
    Patch(color='k', label='Both'),
    Patch(color='r', label='Missing'),
    Patch(color='b', label='Added')
])
plt.savefig(os.path.join(PLT_FOLDER, 'recovered_targets.png'))

plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    # plt.plot(nsamples_list, learned_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    # plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Mean symmetric difference in recovered targets')
plt.legend(handles=[
    *marker_handles,
    Patch(color='k', label='Both'),
    Patch(color='r', label='Missing'),
    Patch(color='b', label='Added')
])
plt.savefig(os.path.join(PLT_FOLDER, 'missing_targets.png'))

plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    # plt.plot(nsamples_list, learned_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='k', marker=marker)
    # plt.plot(nsamples_list, missing_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='r', marker=marker)
    plt.plot(nsamples_list, added_intervention_array.mean(dim='dag').sel(num_unknown=num_unknown), color='b', marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Mean symmetric difference in recovered targets')
plt.legend(handles=[
    *marker_handles,
    Patch(color='k', label='Both'),
    Patch(color='r', label='Missing'),
    Patch(color='b', label='Added')
])
plt.savefig(os.path.join(PLT_FOLDER, 'added_targets.png'))