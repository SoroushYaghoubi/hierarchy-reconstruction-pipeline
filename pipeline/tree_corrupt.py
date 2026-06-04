import random
import argparse
from tree_io import read_closure, write_closure

def corrupt_missing(edges, rate):
    n = int(len(edges) * rate)
    to_remove = set(random.sample(range(len(edges)), n))
    return [e for i, e in enumerate(edges) if i not in to_remove]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stem', help='Tree stem name')
    parser.add_argument('-rate', type=float, default=0.1)
    args = parser.parse_args()

    edges = read_closure(args.stem)
    print(f"Original edges: {len(edges)}")

    corrupted = corrupt_missing(edges, args.rate)
    print(f"Corrupted edges: {len(corrupted)}")

    output_stem = f"{args.stem}_missing_{int(args.rate * 100)}"
    write_closure(corrupted, output_stem)
    print(f"Saved to pipeline/artefacts/{output_stem}.csv")