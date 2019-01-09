from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig
from causaldag import GaussIntervention

DATASET_NAME = 'fig1'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=10,
    nneighbors_list=[1.5],
    ngraphs=50
)

sample_config = SampleConfig(
    dataset_namne=DATASET_NAME,
    nsamples_list=[500],
    ntargets_list=[(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)],
    nsettings_list=[1],
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


