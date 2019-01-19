import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test')
args = parser.parse_args()
print(args.test)

