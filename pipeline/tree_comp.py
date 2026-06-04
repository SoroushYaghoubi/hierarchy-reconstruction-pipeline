import argparse
from tree_io import read_tree

def get_path_to_root(tree, node):
    parent_map = {}
    for parent, children in tree.items():
        for child in children:
            parent_map[child] = parent

    path = []
    current = node
    while current in parent_map:
        path.append(current)
        current = parent_map[current]
    path.append(current)
    return list(reversed(path))

def lca_distance(path_a, path_b):
    lca_depth = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            lca_depth += 1
        else:
            break
    dist_a = len(path_a) - lca_depth
    dist_b = len(path_b) - lca_depth
    return dist_a + dist_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original', help='Original tree stem')
    parser.add_argument('compared', help='Compared tree stem')
    args = parser.parse_args()

    original_tree = read_tree(args.original)
    compared_tree = read_tree(args.compared)

    all_nodes = set(original_tree.keys())

    lca_distances = []
    exact_matches = 0
    depth_errors = []

    for node in all_nodes:
        if node not in compared_tree:
            continue
        path_orig = get_path_to_root(original_tree, node)
        path_comp = get_path_to_root(compared_tree, node)

        dist = lca_distance(path_orig, path_comp)
        lca_distances.append(dist)

        if path_orig == path_comp:
            exact_matches += 1

        depth_errors.append(abs(len(path_orig) - len(path_comp)))

    n = len(lca_distances)
    print(f"Nodes compared: {n}")
    print(f"Exact path matches: {exact_matches} / {n} ({100*exact_matches/n:.1f}%)")
    print(f"Mean LCA distance: {sum(lca_distances)/n:.3f}")
    print(f"Mean depth error: {sum(depth_errors)/n:.3f}")
    print(f"Max LCA distance: {max(lca_distances)}")