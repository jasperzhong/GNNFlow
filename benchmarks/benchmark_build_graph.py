import argparse
import time

from gnnflow.config import get_default_config
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--mem-resource-type", type=str,
                    choices=["cuda", "unified", "pinned"],
                    default="cuda", help="memory resource type")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    build_start = time.time()
    _, dataset_config = get_default_config("TGN", args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config, dataset_df=df)
    build_end = time.time()
    build_time = build_end - build_start
    print('build graph time: {:.4f}s with mem_resource_type={}'.format(
        build_time, args.mem_resource_type))


if __name__ == "__main__":
    main()
