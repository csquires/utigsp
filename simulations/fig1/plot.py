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

NAME = 'fig1'
PLT_FOLDER = os.path.join(PROJECT_FOLDER, 'simulations', 'figures', NAME)
os.makedirs(PLT_FOLDER, exist_ok=True)

nnodes = 10
nneighbors = 1.5
ndags = 50
dag_str = 'nnodes=%d_nneighbors=%s_ndags=%d' % (nnodes, nneighbors, ndags)

nsamples_list = [100, 300, 500]
nsettings_list = [5]
ntargets_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
intervention = 'perfect1'

coords = {
    'nsamples': nsamples_list,
    'num_unknown': [0, 1, 2, 3],
    'dag': list(range(ndags))
}
shd_array_gies = utils.empty_array(coords)
imec_array_gies = utils.empty_array(coords)
shd_array_icp = utils.empty_array(coords)
imec_array_icp = utils.empty_array(coords)
for nsamples, nsettings, (num_known, num_unknown) in itr.product(nsamples_list, nsettings_list, ntargets_list):
    setting_str = f'nsamples={nsamples},num_known={num_known},num_unknown={num_unknown},nsettings={nsettings},intervention={intervention}'
    loc = dict(nsamples=nsamples, num_unknown=num_unknown)

    # === LOAD GIES RESULTS
    gies_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'gies', 'lambda_=%.2e' % 1)
    shd_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'shds.txt'))
    imec_array_gies.loc[loc] = np.loadtxt(os.path.join(gies_results_folder, 'imec.txt'))

    # === LOAD ICP RESULTS
    icp_results_folder = os.path.join(PROJECT_FOLDER, 'simulations', 'results', dag_str, setting_str, 'icp', 'alpha=%.2e' % .01)
    shd_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'shds.txt'))
    imec_array_icp.loc[loc] = np.loadtxt(os.path.join(icp_results_folder, 'imec.txt'))

# === CREATE HANDLES
marker_handles = create_marker_handles([0, 1, 2, 3])

# === PLOT SHDS
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, shd_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    plt.plot(nsamples_list, shd_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('SHD')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.savefig(os.path.join(PLT_FOLDER, 'shd.png'))

# === PLOT PROPORTIONS I-MEC
plt.clf()
for num_unknown, marker in zip([0, 1, 2, 3], MARKERS):
    plt.plot(nsamples_list, imec_array_gies.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['gies'], marker=marker)
    plt.plot(nsamples_list, imec_array_icp.mean(dim='dag').sel(num_unknown=num_unknown), color=ALGS2COLORS['icp'], marker=marker)
plt.xticks(nsamples_list)
plt.xlabel('Number of samples')
plt.ylabel('Proportion I-Markov equivalent to true DAG')
plt.legend(handles=[
    *marker_handles,
    *ALG_HANDLES
])
plt.savefig(os.path.join(PLT_FOLDER, 'imec.png'))

