import argparse
import causaldag as cd
import yaml
import os
from config import DATA_FOLDER
import numpy as np

# == PARSER SETUP
parser = argparse.ArgumentParser('Create DAGs with given parameters')
parser.add_argument('--folder', type=str, help='The name of the folder inside /data where the DAGs will be saved')
parser.add_argument('--ndags', '-d', type=int, help='Number of DAGs to be generated')
parser.add_argument('--nnodes', '-p', type=int, help='Number of nodes in each DAG')
parser.add_argument('--density', '-s', type=float, help='Probability of an edge being present')
parser.add_argument('--epsilon', default=.25, type=float, help='Bounding of edge weights away from 0')

# == PARSE ARGUMENTS AND SAVE
args = parser.parse_args()
nnodes = args.nnodes
ndags = args.ndags
density = args.density
epsilon = args.epsilon
dataset_folder = os.path.join(DATA_FOLDER, args.folder)
os.makedirs(dataset_folder, exist_ok=True)
yaml.dump(vars(args), open(os.path.join(dataset_folder, 'dag_settings.yaml'), 'w'), default_flow_style=False)


# == HELPER FUNCTIONS
def dag2gdag(dag, random_weight):
    return cd.GaussDAG(nodes=list(range(nnodes)), arcs={(i, j): random_weight() for (i, j) in dag.arcs})


def rand_away_zero():
    return (2*np.random.binomial(1, .5) - 1) * np.random.uniform(epsilon, 1)


# == GENERATE DAGS AND SAVE
dags = cd.rand.directed_erdos(nnodes, density, ndags)
gdags = [dag2gdag(dag, rand_away_zero) for dag in dags]
dag_folder = os.path.join(dataset_folder, 'dags')
os.makedirs(dag_folder, exist_ok=True)
for i, gdag in enumerate(gdags):
    gdag_amat = gdag.to_amat()
    np.savetxt(os.path.join(dag_folder, '%d.txt' % i), gdag_amat)
