import os
import causaldag as cd
import numpy as np
from config import DATA_FOLDER
import utils
from dataclasses import asdict


def load_true_and_estimated(dag_config, setting_config, alg_config):
    dag_setting2graph = {}
    dataset_name = dag_config.dataset_name
    for dag_setting in dag_config.settings_list:
        graphs = []
        for i in range(dag_config.ngraphs):
            dag_folder = os.path.join(DATA_FOLDER, dataset_name, 'dags', str(dag_setting), 'dag%d' % i)
            true_dag = cd.DAG.from_amat(np.loadtxt(os.path.join(dag_folder, 'amat.txt')))

            estimated_dags = {ss: {} for ss in setting_config.settings_list}
            for sample_setting in setting_config.settings_list:
                sample_setting_folder = os.path.join(dag_folder, 'samples', str(sample_setting))
                for alg_setting in alg_config.settings_list:
                    est_dag_filename = os.path.join(sample_setting_folder, 'estimates', alg_setting.alg, str(alg_setting) + '.txt')
                    est_dag = cd.DAG.from_amat(np.loadtxt(est_dag_filename))
                    estimated_dags[sample_setting][alg_setting] = est_dag

            graphs.append({'true': true_dag, 'estimated': estimated_dags})
        dag_setting2graph[dag_setting] = graphs
    return dag_setting2graph


def get_shd_array(dag_config, sample_config, alg_config, dag_setting2graph):
    all_nsamples, all_ntargets, all_nsettings = sample_config.settings_to_lists()
    all_nneighbors = dag_config.settings_to_lists()
    shared_coords = {
        'dag': list(range(dag_config.ngraphs)),
        'nsamples': all_nsamples,
        'ntargets': all_ntargets,
        'nsettings': all_nsettings,
        'nneighbors': all_nneighbors
    }
    shd_array_dict = {}
    for alg, settings in alg_config.algs2settings().items():
        shd_array_dict[alg] = utils.empty_array({**shared_coords, **settings})
    for dag_setting, graphs in dag_setting2graph.items():
        for dag_num, dag_dict in enumerate(graphs):
            true_dag = dag_dict['true']
            for sample_setting, alg_setting2dag in dag_dict['estimated'].items():
                for alg_setting, estimated_dag in alg_setting2dag.items():
                    shd_array_dict[alg_setting.alg].loc[dict(
                        dag=dag_num,
                        nsamples=sample_setting.nsamples,
                        ntargets=utils.tup2str(sample_setting.ntargets),
                        nsettings=sample_setting.nsettings,
                        nneighbors=dag_setting.nneighbors,
                        **asdict(alg_setting)
                    )] = true_dag.shd(estimated_dag)

    return shd_array_dict

