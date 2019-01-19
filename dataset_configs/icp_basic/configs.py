from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig, DagSetting, SampleSetting
from dataset_configs.config_types import ICPSetting
from causaldag import ScalingIntervention
import itertools as itr

DATASET_NAME = 'icp_basic'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=3,
    settings_list=[DagSetting(nneighbors=1.5)],
    ngraphs=30
)
sample_config = SampleConfig(
    settings_list=[
        SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
        for nsamples, ntargets, nsettings in itr.product([100, 200, 300], [(1, 0)], [1, 3])
    ],
    intervention=ScalingIntervention(.1, .2),
    dag_config=dag_config
)

icp_settings = [
    ICPSetting(alpha=alpha)
    for alpha in [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1]
]
alg_config = AlgConfig(
    settings_list=icp_settings,
    dag_config=dag_config,
    sample_config=sample_config,
)

if __name__ == '__main__':
    dag_config.save_graphs()
    dags = dag_config.load_graphs()
    sample_config._save_samples()
    results = alg_config.run_alg()

