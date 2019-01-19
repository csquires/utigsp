import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=float)
args = parser.parse_args()
np.savetxt('%s.txt' % args.test, np.array([args.test]))

