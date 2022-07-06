import argparse

import time

from dgnn.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
args = parser.parse_args()

def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    build_start = time.time()
    dgraph = build_dynamic_graph(df)
    build_end = time.time()
    build_time = build_end - build_start
    print('build graph time: {:.4f}s'.format(build_time))

if __name__ == "__main__":
    main()
