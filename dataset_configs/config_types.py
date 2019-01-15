from config import DATA_FOLDER
from utils import tup2str
from R_algs.wrappers import run_gies, run_icp

import os
import random
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, NewType, Union
import itertools as itr
from tqdm import tqdm
import multiprocessing
from functools import partial

import causaldag as cd
from causaldag import ScalingIntervention, SoftInterventionalDistribution
from causaldag.inference.structural import unknown_target_igsp, igsp
from causaldag.utils.ci_tests import hsic_invariance_test, gauss_ci_test

np.random.seed(1729)
random.seed(1729)
# kci_no_regress = partial(kci_invariance_test, regress=False)


@dataclass(frozen=True)
class DagSetting:
    nneighbors: int

    def __str__(self):
        return 'nneighbors=%.2f' % self.nneighbors


@dataclass(frozen=True)
class SampleSetting:
    nsamples: int
    ntargets: Tuple[int, int]
    nsettings: int

    def __str__(self):
        return 'nsamples=%d,ntargets=%s,nsettings=%s' % (self.nsamples, tup2str(self.ntargets), self.nsettings)


@dataclass(frozen=True)
class IGSPSetting:
    nruns: int
    depth: int
    alpha: float
    alpha_invariant: float

    @property
    def alg(self):
        return 'igsp'

    def __str__(self):
            return 'nruns=%d,depth=%d,alpha=%.2e,alpha_inv=%.2e' % (self.nruns, self.depth, self.alpha, self.alpha_invariant)


@dataclass(frozen=True)
class UTIGSPSetting:
    nruns: int
    depth: int
    alpha: float
    alpha_invariant: float

    @property
    def alg(self):
        return 'utigsp'

    def __str__(self):
            return 'nruns=%d,depth=%d,alpha=%.2e,alpha_inv=%.2e' % (self.nruns, self.depth, self.alpha, self.alpha_invariant)


@dataclass(frozen=True)
class GIESSetting:
    lambda_: float

    @property
    def alg(self):
        return 'gies'

    def __str__(self):
        return 'lambda=%.2e' % self.lambda_
    

@dataclass(frozen=True)
class ICPSetting:
    alpha: float

    @property
    def alg(self):
        return 'icp'

    def __str__(self):
        return 'alpha=%.2e' % self.alpha


AlgSetting = NewType('AlgSetting', Union[IGSPSetting, UTIGSPSetting, GIESSetting, ICPSetting])


@dataclass
class DagConfig:
    dataset_name: str
    nnodes: int
    settings_list: List[DagSetting]
    ngraphs: int

    def settings_to_lists(self):
        all_nneighbors = set()
        for settings in self.settings_list:
            all_nneighbors.add(settings.nneighbors)
        return list(sorted(all_nneighbors))

    @property
    def setting_folders(self):
        base_folder = os.path.join(DATA_FOLDER, self.dataset_name, 'dags')
        return {setting: os.path.join(base_folder, str(setting)) for setting in self.settings_list}

    @property
    def setting2dag_folders(self):
        setting2dag_folders = {
            setting: [os.path.join(setting_folder, 'dag%d' % i) for i in range(self.ngraphs)]
            for setting, setting_folder in self.setting_folders.items()
        }
        for dag_folders in setting2dag_folders.values():
            for folder in dag_folders:
                os.makedirs(folder, exist_ok=True)
        return setting2dag_folders

    @property
    def dag_filenames(self):
        return {
            setting: [os.path.join(dag_folder, 'amat.txt') for dag_folder in dag_folders]
            for setting, dag_folders in self.setting2dag_folders.items()
        }

    def save_graphs(self):
        for setting, dag_filenames in self.dag_filenames.items():
            dags = cd.rand.directed_erdos(self.nnodes, setting.nneighbors/(self.nnodes - 1), size=self.ngraphs)
            gdags = [cd.rand.rand_weights(d) for d in dags]
            for dag_filename, gdag in zip(dag_filenames, gdags):
                np.savetxt(dag_filename, gdag.to_amat())

    def load_graphs(self):
        setting2graphs = {}
        for setting, dag_filenames in self.dag_filenames.items():
            dags = [cd.GaussDAG.from_amat(np.loadtxt(dag_fn)) for dag_fn in dag_filenames]
            setting2graphs[setting] = dags
        return setting2graphs


@dataclass
class SampleConfig:
    settings_list: List[SampleSetting]
    dag_config: DagConfig
    intervention: SoftInterventionalDistribution

    def __post_init__(self):
        self.samples = None
        
    def settings_to_lists(self):
        all_nsamples = set()
        all_ntargets = set()
        all_nsettings = set()
        for settings in self.settings_list:
            all_nsamples.add(settings.nsamples)
            all_ntargets.add(settings.ntargets)
            all_nsettings.add(settings.nsettings)
        return list(sorted(all_nsamples)), list(map(tup2str, sorted((all_ntargets)))), list(sorted(all_nsettings))

    def _save_samples(self, verbose=True):
        if verbose: print("Saving samples")
        nodes_list = list(range(self.dag_config.nnodes))
        dag_setting2graph2sample_settings2samples_dict = {}
        for dag_setting, graphs in self.dag_config.load_graphs().items():  # DO THIS FIRST SO WE LOAD EACH GRAPH ONCE
            graph2sample_settings2samples_dict = []
            for g, dag_folder in zip(graphs, self.dag_config.setting2dag_folders[dag_setting]):
                sample_settings2samples_dict = {}
                for ss in self.settings_list:
                    # === RANDOMLY PICK INTERVENTION NODES AND WHICH ARE KNOWN
                    nknown, n_unknown = ss.ntargets
                    known_iv_nodes_list = random.sample(list(itr.combinations(nodes_list, nknown)), ss.nsettings)
                    unknown_iv_nodes_list = [
                        random.sample(set(nodes_list) - set(known_iv_nodes), n_unknown)
                        for known_iv_nodes in known_iv_nodes_list
                    ]

                    # === CREATE FOLDER TO SAVE SAMPLES
                    sample_folder = os.path.join(dag_folder, 'samples', str(ss))
                    iv_sample_folder = os.path.join(sample_folder, 'interventional')
                    os.makedirs(iv_sample_folder, exist_ok=True)

                    # === SAVE SAMPLES
                    obs_samples = g.sample(ss.nsamples)
                    samples_dict = {frozenset(): obs_samples}
                    np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
                    for known_iv_nodes, unknown_iv_nodes in zip(known_iv_nodes_list, unknown_iv_nodes_list):
                        interventions = {iv_node: self.intervention for iv_node in list(known_iv_nodes) + unknown_iv_nodes}
                        samples = g.sample_interventional_soft(interventions, ss.nsamples)
                        samples_dict[frozenset(known_iv_nodes)] = samples
                        known_iv_str = ','.join(map(str, known_iv_nodes))
                        unknown_iv_str = ','.join(map(str, unknown_iv_nodes))
                        iv_str = 'known_ivs=%s;unknown_ivs=%s.txt' % (known_iv_str, unknown_iv_str)
                        np.savetxt(os.path.join(iv_sample_folder, iv_str), samples)
                    sample_settings2samples_dict[SampleSetting(nsamples=ss.nsamples, ntargets=ss.ntargets, nsettings=ss.nsettings)] = samples_dict
                graph2sample_settings2samples_dict.append(sample_settings2samples_dict)
            dag_setting2graph2sample_settings2samples_dict[dag_setting] = graph2sample_settings2samples_dict

        return dag_setting2graph2sample_settings2samples_dict

    def _load_samples(self, verbose=True):
        if verbose: print("Loading samples")
        dag_setting2graph2sample_settings2samples_dict = {}
        for dag_setting, dag_folders in self.dag_config.setting2dag_folders.items():
            graph2sample_settings2samples_dict = []
            for dag_folder in dag_folders:
                sample_settings2samples_dict = {}
                for ss in self.settings_list:
                    sample_folder = os.path.join(dag_folder, 'samples', str(ss))
                    iv_sample_folder = os.path.join(sample_folder, 'interventional')

                    # === LOAD SAMPLES INTO DICTIONARY
                    samples_dict = {frozenset(): np.loadtxt(os.path.join(sample_folder, 'observational.txt'))}
                    for fn in os.listdir(iv_sample_folder):
                        known_ivs = frozenset(map(int, fn.split(';')[0].split('=')[1].split(',')))
                        samples_dict[known_ivs] = np.loadtxt(os.path.join(iv_sample_folder, fn))
                    sample_settings2samples_dict[ss] = samples_dict
                graph2sample_settings2samples_dict.append(sample_settings2samples_dict)
            dag_setting2graph2sample_settings2samples_dict[dag_setting] = graph2sample_settings2samples_dict
        return dag_setting2graph2sample_settings2samples_dict

    def get_samples(self):
        if self.samples is not None:
            return self.samples
        dag_folder = list(self.dag_config.setting2dag_folders.values())[0][0]
        if os.path.exists(os.path.join(dag_folder, 'samples')):
            self.samples = self._load_samples()
        else:
            self.samples = self._save_samples()
        return self.samples


def _run_alg_graph(tup):
    alg_settings, sample_settings, dag_folder = tup
    sample_setting2results = {}
    for ss in sample_settings:
        # === LOAD SAMPLES INTO DICTIONARY
        sample_folder = os.path.join(dag_folder, 'samples', str(ss))
        iv_sample_folder = os.path.join(sample_folder, 'interventional')
        samples_dict = {frozenset(): np.loadtxt(os.path.join(sample_folder, 'observational.txt'))}
        for fn in os.listdir(iv_sample_folder):
            known_ivs = frozenset(map(int, fn.split(';')[0].split('=')[1].split(',')))
            samples_dict[known_ivs] = np.loadtxt(os.path.join(iv_sample_folder, fn))

        # === BUILD SUFFSTAT
        corr = np.corrcoef(samples_dict[frozenset()], rowvar=False)
        nnodes = samples_dict[frozenset()].shape[1]
        suffstat = dict(C=corr, n=ss.nsamples)

        alg_setting2results = {}
        for alg_setting in alg_settings:
            if isinstance(alg_setting, UTIGSPSetting):
                est_dag = unknown_target_igsp(
                    samples_dict,
                    suffstat,
                    nnodes,
                    gauss_ci_test,
                    hsic_invariance_test,
                    alpha=alg_setting.alpha,
                    alpha_invariance=alg_setting.alpha_invariant,
                    depth=alg_setting.depth,
                    nruns=alg_setting.nruns
                )
            elif isinstance(alg_setting, IGSPSetting):
                est_dag = igsp(
                    samples_dict,
                    suffstat,
                    nnodes,
                    gauss_ci_test,
                    hsic_invariance_test,
                    alpha=alg_setting.alpha,
                    alpha_invariance=alg_setting.alpha_invariant,
                    depth=alg_setting.depth,
                    nruns=alg_setting.nruns
                )
            elif isinstance(alg_setting, GIESSetting):
                est_dag = run_gies(
                    sample_folder,
                    lambda_=alg_setting.lambda_
                )
            elif isinstance(alg_setting, ICPSetting):
                est_dag = run_icp(
                    sample_folder
                )
            else:
                raise ValueError('alg must be one of utigsp, igsp, icp, or gies')
            alg_setting2results[alg_setting] = est_dag
        sample_setting2results[ss] = alg_setting2results
    return sample_setting2results


@dataclass
class AlgConfig:
    settings_list: List[AlgSetting]
    dag_config: DagConfig
    sample_config: SampleConfig

    def algs2settings(self):
        d = {'igsp': defaultdict(set), 'utigsp': defaultdict(set), 'gies': defaultdict(set), 'icp': defaultdict(set)}
        for setting in self.settings_list:
            if setting.alg == 'igsp' or setting.alg == 'utigsp':
                d[setting.alg]['nruns'].add(setting.nruns)
                d[setting.alg]['depth'].add(setting.depth)
                d[setting.alg]['alpha'].add(setting.alpha)
                d[setting.alg]['alpha_invariant'].add(setting.alpha_invariant)
            if setting.alg == 'gies':
                d[setting.alg]['lambda_'].add(setting.lambda_)
            if setting.alg == 'icp':
                raise NotImplementedError
        d = {alg: {param: list(vals) for param, vals in settings.items()} for alg, settings in d.items()}
        return d

    def run_alg(self):
        print("Running algorithm")
        dag_setting2results = {}
        for dag_setting, dag_folders in self.dag_config.setting2dag_folders.items():
            # DO IT THIS WAY SO WE CAN HAVE A PROGRESS BAR FOR THE RESULTS
            args = zip(itr.repeat(self.settings_list), itr.repeat(self.sample_config.settings_list), dag_folders)
            with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as POOL:
                dag_setting2results[dag_setting] = list(tqdm(POOL.imap(_run_alg_graph, args), total=len(dag_folders)))

        for dag_setting, graph2sample_setting2alg_setting2results in dag_setting2results.items():
            for i, sample_setting2alg_setting2results in enumerate(graph2sample_setting2alg_setting2results):
                for sample_setting, alg_setting2results in sample_setting2alg_setting2results.items():
                    for alg_setting, dag in alg_setting2results.items():
                        if not isinstance(alg_setting, (GIESSetting, ICPSetting)):  # THESE HAVE ALREADY BEEN SAVED BY R
                            filename = os.path.join(
                                self.dag_config.setting2dag_folders[dag_setting][i],
                                'samples',
                                str(sample_setting),
                                'estimates',
                                alg_setting.alg,
                                str(alg_setting) + '.txt'
                            )
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            np.savetxt(filename, dag.to_amat())
        return dag_setting2results


if __name__ == '__main__':
    test_dag_config = DagConfig(
        dataset_name='test',
        nnodes=8,
        settings_list=[DagSetting(nneighbors=1.5)],
        ngraphs=10
    )
    test_dag_config.save_graphs()
    test_dags = test_dag_config.load_graphs()

    test_sample_config = SampleConfig(
        dataset_name='test',
        settings_list=[
            SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
            for nsamples, ntargets, nsettings in itr.product([100], [(1, 1)], [1])
        ],
        intervention=ScalingIntervention(.1),
        dag_config=test_dag_config
    )
    test_sample_config._save_samples()

    igsp_settings = [
        IGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
        for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
    ]
    utigsp_settings = [
        UTIGSPSetting(nruns=10, depth=4, alpha=alpha, alpha_invariant=alpha_invariant)
        for alpha, alpha_invariant in itr.product([1e-5], [1e-5])
    ]
    test_alg_config = AlgConfig(
        dataset_name='test',
        settings_list= igsp_settings + utigsp_settings,
        dag_config=test_dag_config,
        sample_config=test_sample_config,
    )
    results = test_alg_config.run_alg()


