from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig, DagSetting, SampleSetting
from dataset_configs.config_types import UTIGSPSetting, IGSPSetting, GIESSetting
from causaldag import ScalingIntervention
import itertools as itr

DATASET_NAME = 'igsp_basic'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=20,
    settings_list=[DagSetting(nneighbors=1.5)],
    ngraphs=100
)
sample_config = SampleConfig(
    settings_list=[
        SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
        for nsamples, ntargets, nsettings in itr.product([100, 200, 300], [(1, 0)], [20])
    ],
    intervention=ScalingIntervention(0, .2),
    dag_config=dag_config
)

igsp_settings = [
    IGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5, 1e-3, 1e-1])
]
alg_config = AlgConfig(
    settings_list=igsp_settings,
    dag_config=dag_config,
    sample_config=sample_config,
)

if __name__ == '__main__':
    dag_config.save_graphs()
    dags = dag_config.load_graphs()
    sample_config._save_samples()
    results = alg_config.run_alg()
