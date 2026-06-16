import argparse
from tree_io import read_closure

def compare_closures(original, compared):
    original = set(original)
    compared = set(compared)

    tp = len(original & compared)
    fp = len(compared - original)
    fn = len(original - compared)

    precision = tp / len(compared) if compared else 0
    recall = tp / len(original) if original else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'original_edges': len(original),
        'compared_edges': len(compared),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def print_metrics(metrics):
    print(f"Original edges:  {metrics['original_edges']}")
    print(f"Compared edges:  {metrics['compared_edges']}")
    print(f"True positives:  {metrics['tp']}")
    print(f"False positives: {metrics['fp']}")
    print(f"False negatives: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original', help='Original closure stem')
    parser.add_argument('compared', help='Compared closure stem')
    args = parser.parse_args()

    original = read_closure(args.original)
    compared = read_closure(args.compared)

    metrics = compare_closures(original, compared)
    print_metrics(metrics)