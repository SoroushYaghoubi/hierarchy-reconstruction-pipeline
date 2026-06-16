import argparse
from tree_io import read_closure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original', help='Original closure stem')
    parser.add_argument('compared', help='Compared closure stem')
    args = parser.parse_args()

    original = set(read_closure(args.original))
    compared = set(read_closure(args.compared))

    tp = len(original & compared)   # correct edges
    fp = len(compared - original)   # false edges added
    fn = len(original - compared)   # true edges missing

    precision = tp / len(compared) if compared else 0
    recall = tp / len(original) if original else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Original edges:  {len(original)}")
    print(f"Compared edges:  {len(compared)}")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")