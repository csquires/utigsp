from dataclasses import dataclass
import causaldag as cd
from causaldag import GaussIntervention, InterventionalDistribution
from config import DATA_FOLDER
import os
import numpy as np
from typing import List, Tuple
import itertools as itr
import random
from collections import namedtuple
from causaldag.inference.structural import unknown_target_igsp
from causaldag.utils.ci_tests import kci_invariance_test, gauss_ci_test
from tqdm import tqdm
import multiprocessing
np.random.seed(1729)
random.seed(1729)
from functools import partial
kci_no_regress = partial(kci_invariance_test, regress=False)


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
        return 'nsamples=%d,ntargets=%s,nsettings=%s' % (self.nsamples, self.ntargets, self.nsettings)


@dataclass(frozen=True)
class AlgSetting:
    nruns: int
    depth: int
    alpha: float
    alpha_invariant: float

    def __str__(self):
        return 'nruns=%d,depth=%d,alpha=%.2e,alpha_inv=%.2e' % (self.nruns, self.depth, self.alpha, self.alpha_invariant)


@dataclass
class DagConfig:
    dataset_name: str
    nnodes: int
    nneighbors_list: List[float]
    ngraphs: int

    @property
    def settings(self):
        return [DagSetting(nneighbors=nneighbors) for nneighbors in self.nneighbors_list]

    @property
    def setting_folders(self):
        base_folder = os.path.join(DATA_FOLDER, self.dataset_name, 'dags')
        return {setting: os.path.join(base_folder, str(setting)) for setting in self.settings}

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
    dataset_name: str
    nsamples_list: List[int]
    ntargets_list: List[Tuple[int, int]]
    nsettings_list: List[int]
    dag_config: DagConfig
    intervention: InterventionalDistribution

    def __post_init__(self):
        self.samples = None

    @property
    def settings(self):
        return [
            SampleSetting(nsamples=nsamples, ntargets=ntargets, nsettings=nsettings)
            for nsamples, ntargets, nsettings
            in itr.product(self.nsamples_list, self.ntargets_list, self.nsettings_list)
        ]

    def _save_samples(self, verbose=True):
        if verbose: print("Saving samples")
        nodes_list = list(range(self.dag_config.nnodes))
        dag_setting2graph2sample_settings2samples_dict = {}
        for dag_setting, graphs in self.dag_config.load_graphs().items():  # DO THIS FIRST SO WE LOAD EACH GRAPH ONCE
            graph2sample_settings2samples_dict = []
            for g, dag_folder in zip(graphs, self.dag_config.setting2dag_folders[dag_setting]):
                sample_settings2samples_dict = {}
                for ss in self.settings:
                    # === RANDOMLY PICK INTERVENTION NODES AND WHICH ARE KNOWN
                    all_iv_nodes_list = [frozenset(random.sample(nodes_list, sum(ss.ntargets))) for _ in range(ss.nsettings)]
                    known_iv_nodes_list = [frozenset(random.sample(all_iv_nodes, ss.ntargets[0])) for all_iv_nodes in all_iv_nodes_list]

                    # === CREATE FOLDER TO SAVE SAMPLES
                    sample_folder = os.path.join(dag_folder, 'samples', str(ss))
                    iv_sample_folder = os.path.join(sample_folder, 'interventional')
                    os.makedirs(iv_sample_folder, exist_ok=True)

                    # === SAVE SAMPLES
                    obs_samples = g.sample(ss.nsamples)
                    samples_dict = {frozenset(): obs_samples}
                    np.savetxt(os.path.join(sample_folder, 'observational.txt'), obs_samples)
                    for all_iv_nodes, known_iv_nodes in zip(all_iv_nodes_list, known_iv_nodes_list):
                        samples = g.sample_interventional({iv_node: self.intervention for iv_node in all_iv_nodes}, ss.nsamples)
                        samples_dict[known_iv_nodes] = samples
                        known_iv_str = ','.join(map(str, sorted(known_iv_nodes)))
                        unknown_iv_str = ','.join(map(str, sorted(all_iv_nodes - known_iv_nodes)))
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
                for ss in self.settings:
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
            est_dag = unknown_target_igsp(
                samples_dict,
                suffstat,
                nnodes,
                gauss_ci_test,
                kci_invariance_test,
                alpha=alg_setting.alpha,
                alpha_invariance=alg_setting.alpha_invariant,
                depth=alg_setting.depth,
                nruns=alg_setting.nruns
            )
            alg_setting2results[alg_setting] = est_dag
        sample_setting2results[ss] = alg_setting2results
    return sample_setting2results


@dataclass
class AlgConfig:
    dataset_name: str
    nruns_list: List[int]
    depth_list: List[int]
    alpha_list: List[float]
    alpha_invariant_list: List[float]
    dag_config: DagConfig
    sample_config: SampleConfig

    @property
    def settings(self):
        return [
            AlgSetting(nruns=nruns, depth=depth, alpha=alpha, alpha_invariant=alpha_invariant)
            for nruns, depth, alpha, alpha_invariant in
            itr.product(self.nruns_list, self.depth_list, self.alpha_list, self.alpha_invariant_list)
        ]

    def run_alg(self):
        print("Running algorithm")
        dag_setting2results = {}
        for dag_setting, dag_folders in self.dag_config.setting2dag_folders.items():
            # DO IT THIS WAY SO WE CAN HAVE A PROGRESS BAR FOR THE RESULTS
            args = zip(itr.repeat(self.settings), itr.repeat(self.sample_config.settings), dag_folders)
            with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as POOL:
                dag_setting2results[dag_setting] = list(tqdm(POOL.imap(_run_alg_graph, args), total=len(dag_folders)))

        estimated_folder = os.path.join(DATA_FOLDER, self.dataset_name, 'estimated_dags')
        for dag_setting, graph2sample_setting2alg_setting2results in dag_setting2results.items():
            for i, sample_setting2alg_setting2results in enumerate(graph2sample_setting2alg_setting2results):
                for sample_setting, alg_setting2results in sample_setting2alg_setting2results.items():
                    for alg_setting, dag in alg_setting2results.items():
                        filename = os.path.join(
                            estimated_folder,
                            str(dag_setting),
                            'dag%d' % i,
                            str(sample_setting),
                            'utigsp',
                            str(alg_setting) + '.txt'
                        )
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        np.savetxt(filename, dag.to_amat())
        return dag_setting2results


if __name__ == '__main__':
    test_dag_config = DagConfig(dataset_name='test', nnodes=8, nneighbors_list=[1], ngraphs=10)
    test_dag_config.save_graphs()
    test_dags = test_dag_config.load_graphs()

    test_sample_config = SampleConfig(
        dataset_name='test',
        nsamples_list=[100],
        ntargets_list=[(1, 0)],
        nsettings_list=[1],
        intervention=GaussIntervention(0, 2),
        dag_config=test_dag_config
    )
    test_sample_config._save_samples()

    test_alg_config = AlgConfig(
        dataset_name='test',
        nruns_list=[10],
        depth_list=[4],
        alpha_list=[.01],
        alpha_invariant_list=[.05],
        dag_config=test_dag_config,
        sample_config=test_sample_config,
    )
    results = test_alg_config.run_alg()


