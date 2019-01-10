from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig
from causaldag import GaussIntervention

DATASET_NAME = 'fig2'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=10,
    nneighbors_list=[1.5],
    ngraphs=50
)

sample_config = SampleConfig(
    dataset_name=DATASET_NAME,
    nsamples_list=[100, 300, 500],
    ntargets_list=[(1, 1)],
    nsettings_list=[3],
    dag_config=dag_config,
    intervention=GaussIntervention(0, 2)
)

alg_config = AlgConfig(
    dataset_name=DATASET_NAME,
    nruns_list=[10],
    depth_list=[4],
    alpha_list=[.01],
    alpha_invariant_list=[.05],
    dag_config=dag_config,
    sample_config=sample_config
)

dag_config.save_graphs()
sample_config.get_samples()
alg_config.run_alg()
