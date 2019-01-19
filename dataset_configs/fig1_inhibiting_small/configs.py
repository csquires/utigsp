from dataset_configs.config_types import DagConfig, SampleConfig, AlgConfig, DagSetting, SampleSetting
from dataset_configs.config_types import UTIGSPSetting, IGSPSetting, GIESSetting, UTIGSP_Pool_Setting, IGSP_Pool_Setting, ICPSetting
from causaldag import ScalingIntervention
import itertools as itr

DATASET_NAME = 'fig1_inhibiting_small'
dag_config = DagConfig(
    dataset_name=DATASET_NAME,
    nnodes=5,
    settings_list=[DagSetting(nneighbors=1.5)],
    ngraphs=50
)
sample_config = SampleConfig(
    settings_list=[
        SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
        for nsamples, ntargets, nsettings in itr.product([100, 300, 500], [(3, 0), (2, 1), (1, 2)], [1, 3])
    ],
    intervention=ScalingIntervention(.1, .2),
    dag_config=dag_config
)

icp_settings = [
    ICPSetting(alpha=alpha)
    for alpha in [5e-2]
]
igsp_settings = [
    IGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
utigsp_settings = [
    UTIGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
igsp_pool_settings = [
    IGSP_Pool_Setting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
utigsp_pool_settings = [
    UTIGSP_Pool_Setting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
    for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
]
gies_settings = [
    GIESSetting(lambda_)
    for lambda_ in [50, 100]
]
alg_config = AlgConfig(
    settings_list=igsp_settings+utigsp_settings+gies_settings+utigsp_pool_settings+igsp_pool_settings+icp_settings,
    # settings_list=icp_settings,
    dag_config=dag_config,
    sample_config=sample_config,
)

# if __name__ == '__main__':
    # dag_config.save_graphs()
    # dags = dag_config.load_graphs()
    # sample_config._save_samples()
    # results = alg_config.run_alg()

