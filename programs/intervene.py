from config import DATA_FOLDER
import argparse
import os
import yaml

# == PARSER SETUP
parser = argparse.ArgumentParser('Intervene K times on a set of DAGs')
parser.add_argument('--folder', type=str, help='The name of the dataset folder in /data')
parser.add_argument('--num-interventions', '-K', type=int, help='Number of distinct interventions to perform')
parser.add_argument('--min_known', type=int, default=1, help='Minimum number of known nodes in each intervention')
parser.add_argument('--max_known', type=int, default=1, help='Maximimum number of known nodes in each intervention')
parser.add_argument('--min_unknown', type=int, help='Minimum number of off-target nodes in an intervention')
parser.add_argument('--max_unknown', type=int, help='Maximum number of off-target nodes in an intervention')
parser.add_argument('--nsamples_obs', type=int, help='Number of observational samples')
parser.add_argument('--nsamples_int', type=int, help='Number of samples from each intervention')
# intervention locations?
#

# == PARSE ARGUMENTS AND SAVE
args = parser.parse_args()
dataset_folder = os.path.join(DATA_FOLDER, args.folder)
yaml.dump(vars(args), open(os.path.join(dataset_folder, 'sample_settings.yaml'), 'w'), default_flow_style=False)

# == GENERATE THE OBSERVATIONAL AND INTERVENTIONAL DATA
