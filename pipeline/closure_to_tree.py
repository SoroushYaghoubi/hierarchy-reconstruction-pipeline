import numpy as np
import scipy.sparse.csgraph as csg

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
