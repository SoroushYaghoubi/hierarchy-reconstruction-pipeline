import os
import torch
import argparse
from tree_io import write_tree, write_closure, compute_closure

ARTEFACTS = os.path.join(os.path.dirname(__file__), 'artefacts')

def angular_dist(u, v):
    return 1 - torch.dot(u, v) / (u.norm() * v.norm())

def poincare_dist(u, v):
    diff_norm_sq = (u - v).norm() ** 2
    denom = (1 - u.norm() ** 2) * (1 - v.norm() ** 2)
    return torch.acosh(1 + 2 * diff_norm_sq / denom)

def recover_tree(embeddings, objects, dist_fn):
    N = len(objects)
    norms = embeddings.norm(dim=1)

    parent_of = {}
    for i, node in enumerate(objects):
        if node == 'root':
            continue
        candidates = [j for j in range(N) if norms[j] < norms[i]]
        if not candidates:
            parent_of[node] = 'root'
            continue
        parent_of[node] = objects[min(candidates, key=lambda j: dist_fn(embeddings[i], embeddings[j]).item())]

    recovered_tree = {node: [] for node in objects}
    for node, parent in parent_of.items():
        recovered_tree[parent].append(node)

    return recovered_tree

def recover_tree_global(embeddings, objects):
    N = len(objects)

    dist_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = poincare_dist(embeddings[i], embeddings[j])

    mean_dists = dist_matrix.mean(dim=1)
    root_idx = mean_dists.argmin().item()
    root = objects[root_idx]

    parent_of = {}
    for i, node in enumerate(objects):
        if i == root_idx:
            continue
        candidates = [j for j in range(N) if j != i and mean_dists[j] < mean_dists[i]]
        if not candidates:
            parent_of[node] = root
            continue
        parent_of[node] = objects[min(candidates, key=lambda j: dist_matrix[i][j].item())]

    recovered_tree = {node: [] for node in objects}
    for node, parent in parent_of.items():
        recovered_tree[parent].append(node)

    return recovered_tree, root

def recover_tree_nn(embeddings, objects):
    N = len(objects)

    dist_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = poincare_dist(embeddings[i], embeddings[j])

    mean_dists = dist_matrix.mean(dim=1)
    root_idx = mean_dists.argmin().item()
    root = objects[root_idx]

    dist_to_root = dist_matrix[:, root_idx]

    parent_of = {}
    for i, node in enumerate(objects):
        if i == root_idx:
            continue
        candidates = [j for j in range(N) if j != i and dist_to_root[j] < dist_to_root[i]]
        if not candidates:
            parent_of[node] = root
            continue
        parent_of[node] = objects[min(candidates, key=lambda j: dist_matrix[i][j].item())]

    recovered_tree = {node: [] for node in objects}
    for node, parent in parent_of.items():
        recovered_tree[parent].append(node)

    return recovered_tree, root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint stem (no path or extension)')
    parser.add_argument('-method', choices=['angular', 'poincare'], default='angular')
    args = parser.parse_args()

    checkpoint_path = os.path.join(ARTEFACTS, args.checkpoint + '.pth.best')
    chkpnt = torch.load(checkpoint_path, map_location='cpu')
    embeddings = chkpnt['embeddings']
    objects = chkpnt['objects']

    print(f"Nodes: {len(objects)}")

    # if args.method == 'global':
    #     recovered_tree, root = recover_tree_global(embeddings, objects)
    #     print(f"Detected root: {root}")
    #     print(f"Root children: {recovered_tree[root]}")
    # elif args.method == 'nn':
    #     recovered_tree, root = recover_tree_nn(embeddings, objects)
    #     print(f"Detected root: {root}")
    #     print(f"Root children: {recovered_tree[root]}")
    # else:
    dist_fn = angular_dist if args.method == 'angular' else poincare_dist
    recovered_tree = recover_tree(embeddings, objects, dist_fn)
    print(f"Root children: {recovered_tree['root']}")

    output_stem = args.checkpoint + f'_recovered_{args.method}'
    # write_tree(recovered_tree, output_stem)
    write_closure(compute_closure(recovered_tree), output_stem)
    print(f"Saved to {output_stem}")

