import os
import causaldag as cd
import numpy as np
from config import DATA_FOLDER
import xarray as xr
import utils


def load_true_and_estimated(dag_config, setting_config, alg_config):
    dag_setting2graph = {}
    dataset_name = dag_config.dataset_name
    for dag_setting in dag_config.settings:
        graphs = []
        for i in range(dag_config.ngraphs):
            true_dag_fn = os.path.join(DATA_FOLDER, dataset_name, 'dags', str(dag_setting), 'dag%d' % i, 'amat.txt')
            true_dag = cd.DAG.from_amat(np.loadtxt(true_dag_fn))

            estimated_dags = {ss: {} for ss in setting_config.settings}
            estimated_dag_folder = os.path.join(DATA_FOLDER, dataset_name, 'estimated_dags', str(dag_setting), 'dag%d' % i)
            for sample_setting in setting_config.settings:
                sample_setting_folder = os.path.join(estimated_dag_folder, str(sample_setting))
                for alg_setting in alg_config.settings:
                    est_dag_filename = os.path.join(sample_setting_folder, 'utigsp', str(alg_setting) + '.txt')
                    est_dag = cd.DAG.from_amat(np.loadtxt(est_dag_filename))
                    estimated_dags[sample_setting][alg_setting] = est_dag

            graphs.append({'true': true_dag, 'estimated': estimated_dags})
        dag_setting2graph[dag_setting] = graphs
    return dag_setting2graph


def get_shd_array(dag_config, sample_config, alg_config, dag_setting2graph):
    coords = {
        'dag': list(range(dag_config.ngraphs)),
        'nneighbors': dag_config.nneighbors_list,
        'nsamples': sample_config.nsamples_list,
        'ntargets': list(map(utils.tup2str, sample_config.ntargets_list)),
        'nsettings': sample_config.nsettings_list,
        'nruns': alg_config.nruns_list,
        'depth': alg_config.depth_list,
        'alpha': alg_config.alpha_list,
        'alpha_invariant': alg_config.alpha_invariant_list
    }
    shd_array = utils.empty_array(coords)
    for dag_setting, graphs in dag_setting2graph.items():
        for dag_num, dag_dict in enumerate(graphs):
            true_dag = dag_dict['true']
            for sample_setting, alg_setting2dag in dag_dict['estimated'].items():
                for alg_setting, estimated_dag in alg_setting2dag.items():
                    shd_array.loc[dict(
                        dag=dag_num,
                        nneighbors=dag_setting.nneighbors,
                        nsamples=sample_setting.nsamples,
                        ntargets=utils.tup2str(sample_setting.ntargets),
                        nsettings=sample_setting.nsettings,
                        nruns=alg_setting.nruns,
                        depth=alg_setting.depth,
                        alpha=alg_setting.alpha,
                        alpha_invariant=alg_setting.alpha_invariant
                    )] = true_dag.shd_skeleton(estimated_dag)

    return shd_array

