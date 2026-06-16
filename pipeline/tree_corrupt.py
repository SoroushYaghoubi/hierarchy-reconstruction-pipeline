import random
import argparse
import numpy as np
import scipy.sparse.csgraph as csg
from tree_io import read_closure, write_closure, compute_closure

# --- corruption functions ---

def corrupt_missing(edges, rate):
    n = int(len(edges) * rate)
    to_remove = set(random.sample(range(len(edges)), n))
    return [e for i, e in enumerate(edges) if i not in to_remove]

def corrupt_false(edges, rate):
    all_nodes = list(set(n for e in edges for n in e))
    edge_set = set(edges)
    n = int(len(edges) * rate)
    false_edges = []
    attempts = 0
    while len(false_edges) < n and attempts < n * 10:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            false_edges.append((u, v))
            edge_set.add((u, v))
        attempts += 1
    return edges + false_edges

def corrupt_shuffled(edges, rate):
    all_nodes = list(set(n for e in edges for n in e))
    n = int(len(edges) * rate)
    to_shuffle = set(random.sample(range(len(edges)), n))
    result = []
    for i, (child, ancestor) in enumerate(edges):
        if i in to_shuffle:
            wrong = random.choice(all_nodes)
            while wrong == child:
                wrong = random.choice(all_nodes)
            result.append((child, wrong))
        else:
            result.append((child, ancestor))
    return result

# --- closure to tree reconstruction baselines ---

def naive_closure_to_tree(edges):
    ancestors = {}
    all_nodes = set()
    for child, ancestor in edges:
        all_nodes.add(child)
        all_nodes.add(ancestor)
        if child not in ancestors:
            ancestors[child] = set()
        ancestors[child].add(ancestor)

    tree = {node: [] for node in all_nodes}
    for node in all_nodes:
        if node not in ancestors:
            continue
        parent = min(ancestors[node], key=lambda a: len(ancestors.get(a, set())))
        tree[parent].append(node)

    return tree

def mst_closure_to_tree(edges):
    all_nodes = sorted(set(n for e in edges for n in e))
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    dist = np.zeros((N, N))
    for child, ancestor in edges:
        i, j = node_idx[child], node_idx[ancestor]
        dist[i][j] = 1
        dist[j][i] = 1

    mst = csg.minimum_spanning_tree(dist).toarray()

    degree = {n: 0 for n in all_nodes}
    for child, ancestor in edges:
        degree[ancestor] += 1

    tree = {n: [] for n in all_nodes}
    for i in range(N):
        for j in range(N):
            if mst[i][j] > 0:
                ni, nj = all_nodes[i], all_nodes[j]
                if degree[ni] >= degree[nj]:
                    tree[ni].append(nj)
                else:
                    tree[nj].append(ni)

    return tree

def custom_closure_to_tree(edges):
    ancestors = {}
    all_nodes = set()
    for child, ancestor in edges:
        all_nodes.add(child)
        all_nodes.add(ancestor)
        if child not in ancestors:
            ancestors[child] = set()
        ancestors[child].add(ancestor)

    root_candidates = [n for n in all_nodes if n not in ancestors or len(ancestors[n]) == 0]
    root = max(root_candidates, key=lambda n: sum(1 for a in ancestors.values() if n in a))

    tree = {node: [] for node in all_nodes}
    for node in sorted(all_nodes, key=lambda n: len(ancestors.get(n, set()))):
        if node == root:
            continue
        node_ancestors = ancestors.get(node, set())
        if not node_ancestors:
            tree[root].append(node)
            continue
        parent = max(node_ancestors, key=lambda a: len(ancestors.get(a, set())))
        tree[parent].append(node)

    return tree

# --- main ---

CORRUPTION_FNS = {
    'missing': corrupt_missing,
    'false': corrupt_false,
    'shuffled': corrupt_shuffled,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stem', help='Tree stem name')
    parser.add_argument('-type', choices=['missing', 'false', 'shuffled'], default='missing')
    parser.add_argument('-rate', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    edges = read_closure(args.stem)
    print(f"Original edges: {len(edges)}")

    corrupted = CORRUPTION_FNS[args.type](edges, args.rate)
    print(f"Corrupted edges: {len(corrupted)}")

    output_stem = f"{args.stem}_{args.type}_{int(args.rate * 100)}"

    write_closure(corrupted, output_stem)

    # naive = naive_closure_to_tree(corrupted)
    # write_closure(compute_closure(naive), output_stem + '_naive')

    # mst = mst_closure_to_tree(corrupted)
    # write_closure(compute_closure(mst), output_stem + '_mst')

    # cstm = custom_closure_to_tree(corrupted)
    # write_closure(compute_closure(cstm), output_stem + '_cstm')

    print(f"Saved to pipeline/artefacts/{output_stem}")