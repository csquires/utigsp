from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig, DagSetting, SampleSetting
from dataset_configs.config_types import UTIGSPSetting, IGSPSetting, GIESSetting
from causaldag import GaussIntervention
import itertools as itr

DATASET_NAME = 'fig1_large'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=10,
    settings_list=[DagSetting(nneighbors=1.5)],
    ngraphs=50
)
sample_config = SampleConfig(
    settings_list=[
        SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
        for nsamples, ntargets, nsettings in itr.product([100, 300, 500], [(3, 0), (2, 1), (1, 2)], [3])
    ],
    intervention=GaussIntervention(1, .01),
    dag_config=dag_config
)

igsp_settings = [
    IGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
utigsp_settings = [
    UTIGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
gies_settings = [
    GIESSetting(lambda_)
    for lambda_ in [50, 100]
]
alg_config = AlgConfig(
    settings_list=igsp_settings+utigsp_settings+gies_settings,
    dag_config=dag_config,
    sample_config=sample_config,
)

if __name__ == '__main__':
    dag_config.save_graphs()
    dags = dag_config.load_graphs()
    sample_config._save_samples()
    results = alg_config.run_alg()

