from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig
from causaldag import GaussIntervention

dag_config = DagConfig(dataset_name='test', nnodes=8, nneighbors_list=[1.5], ngraphs=10)
sample_config = SampleConfig(
    dataset_name='test',
    nsamples_list=[500],
    ntargets_list=[(1, 1)],
    nsettings_list=[1],
    intervention=GaussIntervention(0, 2),
    dag_config=dag_config
)
alg_config = AlgConfig(
    dataset_name='test',
    nruns_list=[10],
    depth_list=[4],
    alpha_list=[.01],
    alpha_invariant_list=[.05],
    dag_config=dag_config,
    sample_config=sample_config,
)

if __name__ == '__main__':
    dag_config.save_graphs()
    dags = dag_config.load_graphs()
    sample_config._save_samples()
    results = alg_config.run_alg()

