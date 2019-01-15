from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig, DagSetting, SampleSetting
from dataset_configs.config_types import UTIGSPSetting, IGSPSetting, GIESSetting
from causaldag import ScalingIntervention
import itertools as itr

DATASET_NAME = 'gies_unknown_ivs'

dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=20,
    settings_list=[DagSetting(nneighbors=1.5)],
    ngraphs=30
)
sample_config = SampleConfig(
    settings_list=[
        SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
        for nsamples, ntargets, nsettings in itr.product([200, 500], [(1, 0), (1, 1), (1, 2), (1, 3)], [1, 2, 3])
    ],
    intervention=ScalingIntervention(.1),
    dag_config=dag_config
)

gies_settings = [
    GIESSetting(lambda_)
    for lambda_ in [50, 100]
]
alg_config = AlgConfig(
    settings_list=gies_settings,
    dag_config=dag_config,
    sample_config=sample_config,
)

if __name__ == '__main__':
    dag_config.save_graphs()
    dags = dag_config.load_graphs()
    sample_config._save_samples()
    results = alg_config.run_alg()

